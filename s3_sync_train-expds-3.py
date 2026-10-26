#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый скрипт скачивания и синхронизации датасета train-expds-3 (Selectel S3).

Объединяет два прежних скрипта (boto3_download_train-expds-2.py и
participant_download_s3_data.py) в один файл с тремя режимами:

  --mode update    (по умолчанию) — ДОКАЧКА НОВОГО. Качаются только файлы,
                   которых нет локально, и файлы, чей размер на сервере
                   изменился. Уже скачанные файлы корректного размера не
                   трогаются вообще (даже с --force).
  --mode download  — СКАЧИВАНИЕ. Обычное полное скачивание: существующие
                   корректные файлы пропускаются, а с --force перекачиваются
                   заново.
  --mode verify    — только ПРОВЕРКА (ничего не качает и не удаляет).

Главное отличие от прежних скриптов — докачка крупного файла «с середины»:
данные пишутся в <файл>.<etag>.<размер>.<версия>.part потоково, поэтому обрыв
сети, перезагрузка или Ctrl+C не приводят к потере уже скачанных гигабайт.
Следующий запуск продолжает файл с того же места (HTTP Range). Имя .part
содержит отпечаток версии объекта, поэтому докачка никогда не «склеит» куски
разных версий файла; устаревшие .part удаляются автоматически.

Прочие возможности:
  * повторные попытки на каждый файл с экспоненциальной задержкой;
  * проверка размера после скачивания; --md5 дополнительно сверяет контрольную
    сумму по ETag объекта — и для файлов, залитых одним куском (ETag = MD5),
    и для больших файлов, залитых частями (пересчитывается составной ETag);
  * автоочистка «осиротевших» .part-файлов и временных файлов старого скрипта
    (<имя>.<8 hex>);
  * многопоточность (--threads), прогресс-строка, лог-файл, CSV-отчёт;
  * ключи можно переопределить аргументами или переменными окружения
    EXPDS_ENDPOINT / EXPDS_BUCKET / EXPDS_ACCESS_KEY / EXPDS_SECRET_KEY.

Примеры:
    python3 s3_sync_train-expds-3.py --local-dir ./train-expds-3
    python3 s3_sync_train-expds-3.py --mode download --local-dir ./train-expds-3 --threads 8
    python3 s3_sync_train-expds-3.py --mode update --folder 001_Гнёздово_Нивленское
    python3 s3_sync_train-expds-3.py --mode verify --local-dir ./train-expds-3 --md5
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import logging
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import boto3
    import urllib3
    from botocore.config import Config
except ImportError:  # pragma: no cover
    sys.exit(
        "Не найден boto3.\n"
        "Установите зависимости:  pip install boto3\n"
        "Готовый интерпретатор в этой системе: ~/.venvs/koz3/bin/python"
    )

# ---------------------------------------------------------------------------
# Значения по умолчанию (переопределяются аргументами или переменными окружения)
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = os.environ.get('EXPDS_ENDPOINT', 'https://s3.ru-7.storage.selcloud.ru:443')
DEFAULT_BUCKET = os.environ.get('EXPDS_BUCKET', 'train-expds-3')
DEFAULT_ACCESS_KEY = os.environ.get('EXPDS_ACCESS_KEY', '2c2e18dd7dd54cb89228405480ae6651')
DEFAULT_SECRET_KEY = os.environ.get('EXPDS_SECRET_KEY', '557bddef38bb4b079f52ba017eb94613')

DEFAULT_LOCAL_DIR = '.'
DEFAULT_LOG_FILE = 's3_sync_train-expds-3.log'
DEFAULT_EXCLUDES = ('Старая', 'Архив')

CHUNK_SIZE = 8 * 1024 * 1024          # порция чтения из сети (8 МиБ)
HASH_BLOCK = 4 * 1024 * 1024          # порция чтения при подсчёте контрольной суммы
RENDER_INTERVAL = 0.2                 # как часто перерисовывать прогресс (сек)
RETRY_BASE_DELAY = 3.0                # базовая задержка перед повтором (сек)
RETRY_MAX_DELAY = 30.0                # максимальная задержка перед повтором (сек)

# Размеры частей, которые встречаются при загрузке в S3 (для пересчёта
# составного ETag больших файлов). Порядок — от самых частых к редким.
CANDIDATE_PART_SIZES_MIB = (10, 8, 5, 16, 32, 64, 4, 20, 25, 50, 100, 128, 256, 2, 1, 3, 6, 12, 24, 48, 96)

# <имя>.<8 hex отпечаток>.<размер>.<версия>.part — временный файл докачки
PART_RE = re.compile(r'\.([0-9a-f]{8})\.(\d+)\.(\d+)\.part$')
# <имя>.<8 hex> — временный файл старого скрипта (boto3 TransferManager)
LEGACY_TMP_RE = re.compile(r'\.[0-9A-Fa-f]{8}$')

logger = logging.getLogger('s3sync')


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def human(num: float) -> str:
    """Байты в читаемый вид."""
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TiB"


def human_time(seconds: float) -> str:
    """Секунды в вид ЧЧ:ММ:СС (или ММ:СС)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def md5_file(path: Path) -> str:
    """MD5 локального файла (потоково, без загрузки в память)."""
    digest = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def multipart_etag_file(path: Path, size: int, parts: int, expected: str) -> str | None:
    """Пересчитать составной ETag файла (как у объекта, залитого частями).

    Перебирает стандартные размеры частей, подходящие под их количество, и
    возвращает совпавший hex-ETag либо None, если ни один вариант не сошёлся.
    """
    for mib in CANDIDATE_PART_SIZES_MIB:
        part_size = mib * 1024 * 1024
        if math.ceil(size / part_size) != parts:
            continue
        digest = hashlib.md5()
        with open(path, 'rb') as fh:
            while True:
                block = fh.read(part_size)
                if not block:
                    break
                digest.update(hashlib.md5(block).digest())
        if digest.hexdigest() == expected:
            return digest.hexdigest()
    return None


def interruptible_sleep(seconds: float, stop_event: threading.Event) -> None:
    """Пауза, которую можно прервать (Ctrl+C)."""
    end = time.time() + seconds
    while time.time() < end:
        if stop_event.is_set():
            return
        time.sleep(min(0.5, max(0.0, end - time.time())))


# ---------------------------------------------------------------------------
# Описание объекта в S3 и задачи на скачивание
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteObj:
    key: str
    size: int
    etag_listed: str = ''         # ETag из list_objects_v2 (для версии .part)
    last_modified: int = 0        # LastModified из list_objects_v2 (epoch)

    @property
    def etag_fp(self) -> str:
        """Короткий отпечаток версии объекта.

        ВНИМАНИЕ: у Selectel list_objects_v2 отдаёт для составных объектов
        «усечённый» ETag, который не является MD5. Он используется только как
        непрозрачный признак версии, а контрольная сумма считается отдельно
        (см. verify_object_checksum).
        """
        hexonly = re.sub(r'[^0-9a-f]', '', self.etag_listed.lower())
        return (hexonly or '0' * 8)[:8]

    def part_name(self, name: str) -> str:
        """Имя временного файла докачки, привязанное к версии объекта."""
        return f"{name}.{self.etag_fp}.{self.size}.{self.last_modified}.part"


@dataclass
class Task:
    obj: RemoteObj
    dest: Path
    action: str                  # new | changed | redo | skip | skip_changed
    offset: int = 0              # сколько байт уже лежит в .part (докачка)
    part: Path | None = None

    @property
    def todo(self) -> bool:
        return self.action in ('new', 'changed', 'redo')


ACTION_LABEL = {
    'new': 'новый',
    'changed': 'изменён',
    'redo': 'перекачать',
    'skip': 'уже есть',
    'skip_changed': 'изменён (пропущен)',
}


# ---------------------------------------------------------------------------
# Прогресс
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self, total_files: int, total_bytes: int, base_files: int, base_bytes: int):
        self.lock = threading.RLock()
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.base_files = base_files
        self.base_bytes = base_bytes
        self.session_bytes = 0
        self.completed = base_files
        self.counts: collections.Counter = collections.Counter()
        self.failures: list[tuple[str, str]] = []
        self.start = time.time()
        self.last_render = 0.0

    @property
    def position(self) -> int:
        return min(self.total_bytes, self.base_bytes + self.session_bytes)

    def add_bytes(self, n: int) -> None:
        with self.lock:
            self.session_bytes += n
        self.render()

    def finish(self, status: str, key: str = '', error: str = '') -> None:
        with self.lock:
            self.completed += 1
            self.counts[status] += 1
            if status == 'failed':
                self.failures.append((key, error))
        self.render(force=True)

    def render(self, force: bool = False) -> None:
        with self.lock:
            now = time.time()
            if not force and (now - self.last_render) < RENDER_INTERVAL:
                return
            self.last_render = now
            elapsed = max(1e-6, now - self.start)
            speed = self.session_bytes / elapsed
            pos = self.position
            pct = (pos / self.total_bytes * 100) if self.total_bytes else 100.0
            eta = (self.total_bytes - pos) / speed if speed > 0 else 0.0
            print(
                f"\r  Прогресс: [{self.completed}/{self.total_files}] {pct:5.1f}% | "
                f"{human(pos)}/{human(self.total_bytes)} | {human(speed)}/s | "
                f"осталось ~{human_time(eta)}   ",
                end='', flush=True,
            )


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def make_client(args):
    if not args.tls_verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return boto3.client(
        's3',
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        verify=args.tls_verify,
        config=Config(
            signature_version='s3v4',
            max_pool_connections=max(10, args.threads + 4),
            connect_timeout=30,
            read_timeout=120,           # большим файлам нужно время между чанками
            retries={'max_attempts': 5, 'mode': 'adaptive'},
        ),
    )


def list_remote(client, args) -> list[RemoteObj]:
    """Список объектов бакета в области видимости (--folder / --exclude)."""
    print(f"Сканирование бакета: {args.bucket}" + (f" / {args.folder}" if args.folder else ""))
    logger.info("Сканирование: bucket=%s prefix=%r", args.bucket, args.folder)

    objects: list[RemoteObj] = []
    skipped_excluded = 0
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.folder):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):                       # маркер папки
                continue
            if args.excludes and any(part in args.excludes for part in key.split('/')):
                skipped_excluded += 1
                logger.info("Пропуск (исключённая папка): %s", key)
                continue
            modified = obj.get('LastModified')
            objects.append(RemoteObj(
                key=key,
                size=int(obj['Size']),
                etag_listed=str(obj.get('ETag', '')),
                last_modified=int(modified.timestamp()) if modified else 0,
            ))
    if skipped_excluded:
        print(f"  Исключено служебных файлов: {skipped_excluded} — папки {', '.join(args.excludes)}")
    return objects


def build_plan(objects: list[RemoteObj], local_dir: Path, args) -> list[Task]:
    """Определить, что делать с каждым объектом."""
    tasks: list[Task] = []
    for obj in objects:
        dest = local_dir / obj.key
        action = 'new'
        if dest.is_file():
            if dest.stat().st_size == obj.size:
                action = 'redo' if (args.mode == 'download' and args.force) else 'skip'
            else:
                action = 'changed'
        if action == 'changed' and args.skip_changed:
            action = 'skip_changed'
        tasks.append(Task(obj=obj, dest=dest, action=action))
    return tasks


def apply_resume_offsets(tasks: list[Task], args) -> None:
    """Найти существующие .part-файлы и учесть уже скачанные байты."""
    for task in tasks:
        if not task.todo:
            continue
        part = task.dest.with_name(task.obj.part_name(task.dest.name))
        task.part = part
        if args.no_resume:
            if part.exists():
                try:
                    part.unlink()
                    logger.info("--no-resume: удалён .part %s", part)
                except OSError as exc:
                    logger.warning("Не удалось удалить %s: %s", part, exc)
            continue
        if part.is_file():
            size = part.stat().st_size
            if size > task.obj.size:            # битый/чужой .part — начинаем заново
                logger.warning("Некорректный .part (больше оригинала), удаляю: %s", part)
                try:
                    part.unlink()
                except OSError as exc:
                    logger.warning("Не удалось удалить %s: %s", part, exc)
            else:
                task.offset = size


def cleanup_stale(remote_fingerprints: set, local_dir: Path, args) -> int:
    """Удалить .part и временные файлы старого скрипта, которые уже нельзя использовать."""
    removed = 0
    if not local_dir.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(local_dir):
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(local_dir).as_posix()

            # Вне области видимости (--folder) ничего не трогаем
            if args.folder and not rel.startswith(args.folder):
                continue

            match = PART_RE.search(name)
            if match:
                base = name[:match.start()]
                base_path = Path(dirpath) / base
                base_rel = base_path.relative_to(local_dir).as_posix()
                fingerprint = (base_rel, match.group(1), int(match.group(2)), int(match.group(3)))
                if fingerprint not in remote_fingerprints or base_path.is_file():
                    try:
                        full.unlink()
                        logger.info("Удалён неактуальный .part: %s", rel)
                        removed += 1
                    except OSError as exc:
                        logger.warning("Не удалось удалить %s: %s", rel, exc)
                continue

            if LEGACY_TMP_RE.search(name) and not name.endswith('.part'):
                try:
                    full.unlink()
                    logger.info("Удалён временный файл старого скрипта: %s", rel)
                    removed += 1
                except OSError as exc:
                    logger.warning("Не удалось удалить %s: %s", rel, exc)
    return removed


def verify_object_checksum(client, args, obj: RemoteObj, path: Path) -> bool | None:
    """Сверить контрольную сумму файла с ETag объекта.

    Возвращает True (совпало), False (не совпало) или None (проверить нельзя).
    ETag берётся через head_object: list_objects_v2 у Selectel возвращает для
    составных объектов некорректный (усечённый) ETag.
    """
    try:
        etag = client.head_object(Bucket=args.bucket, Key=obj.key)['ETag'].strip('"')
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("Не удалось получить ETag для %s: %s", obj.key, exc)
        return None

    if '-' in etag:
        hexpart, _, parts_txt = etag.rpartition('-')
        if len(hexpart) != 32 or not parts_txt.isdigit() or int(parts_txt) < 1:
            return None
        expected = hexpart.lower()
        return multipart_etag_file(path, path.stat().st_size, int(parts_txt), expected) == expected

    if len(etag) == 32:
        return md5_file(path) == etag.lower()

    return None


# ---------------------------------------------------------------------------
# Скачивание одного файла (с докачкой)
# ---------------------------------------------------------------------------
def download_object(client, task: Task, args, stats: Stats,
                    stop_event: threading.Event) -> tuple[str, str]:
    """Скачать/докачать один объект. Возвращает (статус, текст ошибки)."""
    obj, dest = task.obj, task.dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return 'failed', f"не удалось создать папку {dest.parent}: {exc}"

    # Битый/устаревший финальный файл нельзя докачивать — удаляем
    if dest.is_file():
        try:
            if dest.stat().st_size != obj.size or task.action == 'redo':
                dest.unlink()
                logger.info("Удалён локальный файл для перезаписи: %s", dest)
        except OSError as exc:
            return 'failed', f"не удалось удалить {dest}: {exc}"

    part = task.part or dest.with_name(obj.part_name(dest.name))

    for attempt in range(1, args.retries + 1):
        if stop_event.is_set():
            return 'interrupted', 'остановлено пользователем'

        offset = part.stat().st_size if part.is_file() else 0
        if offset > obj.size:
            try:
                part.unlink()
            except OSError:
                pass
            offset = 0

        # Файл уже докачан полностью (например, обрыв случился до переименования)
        if offset == obj.size:
            try:
                os.replace(part, dest)
                logger.info("Докачка завершена: %s", obj.key)
                return 'ok', ''
            except OSError as exc:
                return 'failed', f"не удалось переименовать {part}: {exc}"

        try:
            logger.info("Скачивание (попытка %d, с %d из %d байт): %s",
                        attempt, offset, obj.size, obj.key)
            kwargs = {'Bucket': args.bucket, 'Key': obj.key}
            if offset:
                kwargs['Range'] = f'bytes={offset}-'
            response = client.get_object(**kwargs)
            status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')

            mode = 'ab' if offset else 'wb'
            if offset and status_code != 206:
                # Сервер проигнорировал Range — начинаем файл заново
                logger.warning("Сервер не поддержал докачку (HTTP %s) для %s — скачиваю заново",
                               status_code, obj.key)
                offset = 0
                mode = 'wb'

            content_length = response.get('ContentLength')
            if offset and content_length is not None and content_length != obj.size - offset:
                raise IOError(f"сервер вернул не тот диапазон: {content_length}, "
                              f"ожидалось {obj.size - offset}")

            body = response['Body']
            try:
                with open(part, mode, buffering=0) as fh:
                    while True:
                        if stop_event.is_set():
                            raise InterruptedError
                        chunk = body.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        offset += len(chunk)
                        stats.add_bytes(len(chunk))
            finally:
                body.close()

            actual = part.stat().st_size
            if actual != obj.size:
                raise IOError(f"не совпал размер: получено {actual}, ожидалось {obj.size}")

            os.replace(part, dest)

            if args.md5:
                checksum_ok = verify_object_checksum(client, args, obj, dest)
                if checksum_ok is False:
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    raise IOError("контрольная сумма (ETag) не совпала")
                if checksum_ok is None:
                    logger.warning("Контрольная сумму для %s проверить не удалось", obj.key)

            logger.info("Скачано успешно: %s", obj.key)
            return 'ok', ''

        except InterruptedError:
            return 'interrupted', 'прервано (Ctrl+C) — файл .part сохранён'
        except Exception as exc:                                # noqa: BLE001
            if stop_event.is_set():
                return 'interrupted', 'остановлено пользователем'
            logger.warning("Ошибка (%s, попытка %d/%d): %s",
                           obj.key, attempt, args.retries, exc)
            if attempt >= args.retries:
                logger.error("Отказ после %d попыток: %s: %s", args.retries, obj.key, exc)
                return 'failed', str(exc)
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY) + random.uniform(0, 2)
            print(f"\n  Повтор {attempt}/{args.retries - 1} для {Path(obj.key).name} "
                  f"через {delay:.1f}с: {exc}", flush=True)
            interruptible_sleep(delay, stop_event)

    return 'failed', 'исчерпаны попытки'


def run_downloads(client, tasks: list[Task], args, stats: Stats,
                  stop_event: threading.Event) -> None:
    todo = [t for t in tasks if t.todo]
    if not todo:
        print("Нечего скачивать — все файлы уже на месте.")
        return

    width = max(len(ACTION_LABEL.get(t.action, t.action)) for t in todo)
    base = Path(args.local_dir)
    print(f"Скачивание {len(todo)} файл(ов) в {args.threads} поток(ов):")
    for task in todo[:15]:
        rel = os.path.relpath(task.dest, base)
        note = f", докачка с {human(task.offset)}" if task.offset else ""
        print(f"  - {ACTION_LABEL.get(task.action, task.action):<{width}}  {rel} "
              f"({human(task.obj.size)}{note})")
    if len(todo) > 15:
        print(f"  ... и ещё {len(todo) - 15} файл(ов)")
    print()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(download_object, client, task, args, stats, stop_event): task
            for task in todo
        }
        try:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    status, error = future.result()
                except Exception as exc:                        # noqa: BLE001
                    logger.exception("Непредвиденная ошибка: %s", task.obj.key)
                    status, error = 'failed', f"внутренняя ошибка: {exc}"
                if status == 'ok':
                    logger.info("Готово: %s", task.obj.key)
                    stats.finish(task.action, task.obj.key)
                elif status == 'interrupted':
                    stats.finish('interrupted', task.obj.key, error)
                else:
                    print(f"\n  ОШИБКА: {task.obj.key}: {error}")
                    stats.finish('failed', task.obj.key, error)
        except KeyboardInterrupt:
            stop_event.set()
            print("\n\nПрерывание по Ctrl+C — останавливаю потоки "
                  "(незавершённые .part-файлы сохраняются для докачки)...")
            raise
        finally:
            if stop_event.is_set():
                executor.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------
# Проверка
# ---------------------------------------------------------------------------
def verify_scope(client, objects: list[RemoteObj], local_dir: Path,
                 args) -> tuple[int, int, int]:
    """Сверить локальные файлы с сервером. Возвращает (ok, missing, mismatch)."""
    print("\nПроверка целостности локальной копии..."
          + (" (с контрольными суммами, это может занять время)" if args.md5 else ""))
    ok = missing = mismatch = unverifiable = 0
    missing_bytes = 0
    problems: list[str] = []
    remote_keys: set[str] = set()

    for obj in objects:
        remote_keys.add(obj.key)
        dest = local_dir / obj.key
        if not dest.is_file():
            missing += 1
            missing_bytes += obj.size
            if len(problems) < 15:
                problems.append(f"  НЕТ:      {obj.key}")
            continue
        size = dest.stat().st_size
        if size != obj.size:
            mismatch += 1
            missing_bytes += obj.size
            if len(problems) < 15:
                problems.append(f"  РАЗМЕР:   {obj.key} (локально {human(size)}, "
                                f"на сервере {human(obj.size)})")
            continue
        if args.md5:
            checksum_ok = verify_object_checksum(client, args, obj, dest)
            if checksum_ok is False:
                mismatch += 1
                if len(problems) < 15:
                    problems.append(f"  СУММА:    {obj.key} (не совпала с ETag объекта)")
                continue
            if checksum_ok is None:
                unverifiable += 1
        ok += 1

    for line in problems:
        print(line)
    if missing + mismatch > len(problems):
        print(f"  ... и ещё {missing + mismatch - len(problems)} проблем(ы)")

    # Лишние локальные файлы (информационно, ничего не удаляем)
    extra = 0
    if local_dir.is_dir() and not args.folder:
        log_path = os.path.abspath(args.log_file)
        rep_path = os.path.abspath(args.report) if args.report else ''
        for dirpath, _dirnames, filenames in os.walk(local_dir):
            for name in filenames:
                full = Path(dirpath) / name
                if os.path.abspath(full) in (log_path, rep_path):
                    continue
                if PART_RE.search(name) or LEGACY_TMP_RE.search(name):
                    continue
                rel = full.relative_to(local_dir).as_posix()
                if rel not in remote_keys:
                    extra += 1
                    if extra <= 5:
                        print(f"  ЛИШНИЙ:   {rel}")

    print("\nСводка проверки:")
    print(f"  Всего объектов в области: {len(objects)}")
    print(f"  Корректных:               {ok}")
    print(f"  Отсутствует:              {missing}"
          + (f" ({human(missing_bytes)})" if missing_bytes else ""))
    print(f"  Не совпал размер/сумма:   {mismatch}")
    if unverifiable:
        print(f"  Сумма не проверялась:     {unverifiable} (нестандартный ETag)")
    if extra:
        print(f"  Лишних локальных файлов:  {extra} (не удаляются)")
    if missing == 0 and mismatch == 0:
        print("  ✓ Все файлы на месте и корректны")
    else:
        print("  ✗ Есть расхождения — запустите режим update, чтобы докачать")
    return ok, missing, mismatch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='s3_sync_train-expds-3.py',
        description='Скачивание, докачка нового и проверка датасета train-expds-3 (Selectel S3).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Примеры:\n'
            '  python3 s3_sync_train-expds-3.py --local-dir ./train-expds-3\n'
            '  python3 s3_sync_train-expds-3.py --mode download --force --threads 8\n'
            '  python3 s3_sync_train-expds-3.py --mode verify --md5\n'
        ),
    )
    parser.add_argument('--mode', choices=('update', 'download', 'verify'), default='update',
                        help='update — докачка нового/изменённого (по умолчанию); '
                             'download — скачивание (с --force перекачивает существующее); '
                             'verify — только проверка')
    parser.add_argument('--folder', '--prefix', dest='folder', default='',
                        help='Скачивать только эту папку/префикс (по умолчанию — весь бакет)')
    parser.add_argument('--local-dir', '--dest', dest='local_dir', default=DEFAULT_LOCAL_DIR,
                        help=f'Куда сохранять (по умолчанию: {DEFAULT_LOCAL_DIR})')
    parser.add_argument('--threads', type=int, default=4,
                        help='Количество параллельных потоков (по умолчанию: 4)')
    parser.add_argument('--force', action='store_true',
                        help='В режиме download перекачивать даже корректные файлы '
                             '(в режиме update игнорируется)')
    parser.add_argument('--skip-changed', action='store_true',
                        help='Не качать файлы, изменившиеся на сервере (только новые)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Не докачивать .part-файлы, начинать каждый файл заново')
    parser.add_argument('--retries', type=int, default=6,
                        help='Попыток на файл при ошибках сети (по умолчанию: 6)')
    parser.add_argument('--md5', action='store_true',
                        help='Дополнительно сверять контрольную сумму с ETag объекта '
                             '(для больших составных файлов ETag пересчитывается локально; '
                             'заметно нагружает диск)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Синоним --mode verify (совместимость со старыми скриптами)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Не проверять результат после скачивания')
    parser.add_argument('--dry-run', action='store_true',
                        help='Только показать план, ничего не скачивать')
    parser.add_argument('--report', default='',
                        help='Путь к CSV-отчёту (по умолчанию не создаётся)')
    parser.add_argument('--log-file', default=DEFAULT_LOG_FILE,
                        help=f'Файл лога (по умолчанию: {DEFAULT_LOG_FILE})')
    parser.add_argument('--exclude', action='append', dest='excludes', metavar='ИМЯ',
                        help='Имя папки для исключения (можно повторять; '
                             f'по умолчанию: {", ".join(DEFAULT_EXCLUDES)})')
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT, help='Endpoint S3')
    parser.add_argument('--bucket', default=DEFAULT_BUCKET, help='Имя бакета')
    parser.add_argument('--access-key', default=DEFAULT_ACCESS_KEY, help='Access key')
    parser.add_argument('--secret-key', default=DEFAULT_SECRET_KEY, help='Secret key')
    parser.add_argument('--tls-verify', action='store_true',
                        help='Проверять TLS-сертификат сервера (по умолчанию проверка отключена)')

    args = parser.parse_args(argv)
    if not args.excludes:
        args.excludes = list(DEFAULT_EXCLUDES)
    if args.verify_only:
        args.mode = 'verify'
    if args.threads < 1:
        parser.error('--threads должен быть >= 1')
    if args.retries < 1:
        parser.error('--retries должен быть >= 1')
    if args.mode == 'update' and args.force:
        print('Примечание: в режиме update --force игнорируется — '
              'существующие файлы не перекачиваются.')
    return args


def setup_logging(args) -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                           # noqa: BLE001
        pass
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    try:
        handler = logging.FileHandler(args.log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    except OSError as exc:
        print(f"Предупреждение: не удалось открыть лог-файл {args.log_file}: {exc}")


def write_report(path: str, tasks: list[Task], statuses: dict, args) -> None:
    if not path:
        return
    try:
        with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
            writer = csv.writer(fh, delimiter=';')
            writer.writerow(['key', 'size_bytes', 'action', 'result', 'error'])
            for task in tasks:
                result, error = statuses.get(task.obj.key, ('', ''))
                writer.writerow([task.obj.key, task.obj.size,
                                 ACTION_LABEL.get(task.action, task.action), result, error])
        print(f"Отчёт: {path}")
    except OSError as exc:
        print(f"Предупреждение: не удалось записать отчёт {path}: {exc}")


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Датасет train-expds-3 | режим: {args.mode.upper()}")
    print(f"Endpoint:  {args.endpoint}")
    print(f"Бакет:     {args.bucket}" + (f" | папка: {args.folder}" if args.folder else ""))
    print(f"Локально:  {local_dir}")
    print(f"Потоки:    {args.threads} | попыток на файл: {args.retries} | "
          f"докачка .part: {'выкл' if args.no_resume else 'вкл'}"
          + (" | контроль сумм: вкл" if args.md5 else ""))
    print(f"Лог:       {args.log_file}")
    print("=" * 80)
    logger.info("Запуск: %s", vars(args))

    try:
        client = make_client(args)
        objects = list_remote(client, args)
    except Exception as exc:                                    # noqa: BLE001
        print(f"ОШИБКА доступа к S3: {exc}")
        logger.error("Ошибка доступа к S3: %s", exc)
        return 1

    if not objects:
        print("В бакете нет объектов по заданному пути.")
        return 1

    total_bytes = sum(o.size for o in objects)
    print(f"Найдено файлов: {len(objects)} ({human(total_bytes)})")

    tasks = build_plan(objects, local_dir, args)

    if args.mode == 'verify':
        print("РЕЖИМ ПРОВЕРКИ (ничего не скачивается и не удаляется)")
        _ok, missing, mismatch = verify_scope(client, objects, local_dir, args)
        logger.info("Проверка завершена: missing=%d mismatch=%d", missing, mismatch)
        return 0 if (missing == 0 and mismatch == 0) else 1

    counters = collections.Counter(t.action for t in tasks)
    to_fetch = [t for t in tasks if t.todo]
    fetch_bytes = sum(t.obj.size - t.offset for t in to_fetch)

    print("План:")
    print(f"  новых:                  {counters['new']}")
    print(f"  изменившихся:           {counters['changed']}")
    if counters['redo']:
        print(f"  на перекачку (--force): {counters['redo']}")
    print(f"  уже на месте (пропуск): {counters['skip']}")
    if counters['skip_changed']:
        print(f"  изменённых, пропущено по --skip-changed: {counters['skip_changed']}")
    print(f"  всего к загрузке:       {len(to_fetch)} файл(ов), ~{human(fetch_bytes)}")

    if args.dry_run:
        print("\n--dry-run: ничего не скачано.")
        return 0

    if not to_fetch:
        if counters['skip_changed']:
            print("\nНет файлов для загрузки: изменившиеся пропущены по --skip-changed.")
        else:
            print("\nВсё уже скачано — синхронизация не требуется.")
        if not args.no_verify:
            verify_scope(client, objects, local_dir, args)
        return 0

    # Уборка мусора и учёт уже скачанных .part
    fingerprints = {(o.key, o.etag_fp, o.size, o.last_modified) for o in objects}
    removed = cleanup_stale(fingerprints, local_dir, args)
    if removed:
        print(f"Удалено неактуальных временных файлов: {removed}")
    apply_resume_offsets(tasks, args)

    base_bytes = sum(t.obj.size for t in tasks if t.action in ('skip', 'skip_changed'))
    base_bytes += sum(t.offset for t in tasks if t.todo)
    base_files = counters['skip'] + counters['skip_changed']
    stats = Stats(total_files=len(tasks), total_bytes=total_bytes,
                  base_files=base_files, base_bytes=base_bytes)
    statuses: dict[str, tuple[str, str]] = {}

    print()
    stop_event = threading.Event()
    interrupted_exit = False
    try:
        run_downloads(client, tasks, args, stats, stop_event)
    except KeyboardInterrupt:
        interrupted_exit = True
    finally:
        stats.render(force=True)
        print()

    if interrupted_exit:
        print("Работа прервана. Запустите скрипт снова — докачка продолжится с места обрыва.")
        logger.info("Прервано пользователем")
        return 130

    for task in tasks:
        if not task.todo:
            statuses[task.obj.key] = ('пропущен', '')
        elif task.action == 'new':
            statuses[task.obj.key] = ('скачан', '')
        elif task.action == 'changed':
            statuses[task.obj.key] = ('обновлён', '')
        else:
            statuses[task.obj.key] = ('перекачан', '')
    for key, error in stats.failures:
        statuses[key] = ('ОШИБКА', error)

    interrupted = bool(stats.counts['interrupted'])
    elapsed = time.time() - stats.start
    speed = stats.session_bytes / elapsed if elapsed > 0 else 0
    print("=" * 80)
    print("Итог:")
    print(f"  Всего в области:        {len(tasks)} файл(ов), {human(total_bytes)}")
    print(f"  Скачано новых:          {stats.counts['new']}")
    print(f"  Обновлено (размер):     {stats.counts['changed']}")
    print(f"  Перекачано (--force):   {stats.counts['redo']}")
    print(f"  Пропущено (уже есть):   {counters['skip'] + counters['skip_changed']}")
    print(f"  Ошибок:                 {stats.counts['failed']}")
    if interrupted:
        print(f"  Прервано:               {stats.counts['interrupted']}")
    print(f"  Принято по сети:        {human(stats.session_bytes)} за {human_time(elapsed)} "
          f"({human(speed)}/s)")
    leftover = [t for t in tasks if t.part and t.part.is_file()]
    if leftover:
        print(f"  Незавершённых файлов:   {len(leftover)} (.part сохранены — "
              f"следующий запуск продолжит докачку)")
    if stats.failures:
        print("  Сбойные файлы:")
        for key, error in stats.failures[:10]:
            print(f"    - {key}: {error}")
        if len(stats.failures) > 10:
            print(f"    ... и ещё {len(stats.failures) - 10}")
    print("=" * 80)
    logger.info("Итог: new=%d changed=%d redo=%d skip=%d failed=%d bytes=%d time=%.1f",
                stats.counts['new'], stats.counts['changed'], stats.counts['redo'],
                counters['skip'] + counters['skip_changed'], stats.counts['failed'],
                stats.session_bytes, elapsed)

    write_report(args.report, tasks, statuses, args)

    result_ok = stats.counts['failed'] == 0 and not interrupted
    if not args.no_verify and stats.counts['failed'] == 0:
        _ok, missing, mismatch = verify_scope(client, objects, local_dir, args)
        result_ok = result_ok and missing == 0 and mismatch == 0

    return 0 if result_ok else 1


if __name__ == '__main__':
    sys.exit(main())
