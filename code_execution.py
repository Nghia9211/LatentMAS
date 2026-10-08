"""Process lifecycle for code-task evaluation. This is NOT a security sandbox."""
import multiprocessing
import traceback


def _worker(sender, code):
    # Module-level target: pickleable with spawn, including after vLLM startup.
    try:
        exec(code, {})
        result = (True, None)
    except BaseException:
        result = (False, traceback.format_exc()[-8192:])
    try:
        sender.send(result)
    finally:
        sender.close()


def run_with_timeout(code, timeout):
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    ctx = multiprocessing.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(sender, code))
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            return False, f'TimeoutError: Execution exceeded {timeout} seconds'
        try:
            return receiver.recv()
        except EOFError:
            return False, 'Worker exited without returning a result'
    finally:
        if process.pid is not None:
            process.join(0.1)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        receiver.close()
        sender.close()
