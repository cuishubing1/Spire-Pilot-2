using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Sts2HumanRecorder;

internal sealed class AsyncRecordWriter : IDisposable
{
    private abstract record WriteItem;
    private sealed record RecordItem(SortedDictionary<string, object?> Record, ActionCommitToken? Commit) : WriteItem;
    private sealed record BarrierItem(bool Durable, TaskCompletionSource<bool> Completion) : WriteItem;
    private sealed record StopItem(TaskCompletionSource<bool> Completion) : WriteItem;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    private readonly BlockingCollection<WriteItem> _queue = new(512);
    private readonly FileStream _stream;
    private readonly StreamWriter _writer;
    private readonly Thread _thread;
    private string _previousHash;
    private Exception? _fault;
    private int _queueHighWatermark;
    private long _recordsWritten;
    private long _bytesWritten;
    private long _serializeTicks;
    private long _flushTicks;
    private bool _disposed;

    public AsyncRecordWriter(string path, FileMode mode, string previousHash)
    {
        _previousHash = previousHash;
        _stream = new FileStream(path, mode, FileAccess.Write, FileShare.Read, 65536,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        _writer = new StreamWriter(_stream, new UTF8Encoding(false), 65536);
        _thread = new Thread(WriterLoop)
        {
            IsBackground = true,
            Name = "HumanRecorder.Writer"
        };
        _thread.Start();
    }

    public Exception? Fault => Volatile.Read(ref _fault);
    public string PreviousHash => _previousHash;

    public bool TryEnqueue(SortedDictionary<string, object?> record, ActionCommitToken? commit = null, int timeoutMs = 100)
    {
        if (_disposed || Fault is not null) return false;
        try
        {
            if (!_queue.TryAdd(new RecordItem(record, commit), timeoutMs)) return false;
            UpdateHighWatermark(_queue.Count);
            return true;
        }
        catch (InvalidOperationException) { return false; }
    }

    public bool Flush(bool durable, int timeoutMs = 5000)
    {
        if (_disposed || Fault is not null) return false;
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            if (!_queue.TryAdd(new BarrierItem(durable, completion), timeoutMs)) return false;
            return completion.Task.Wait(timeoutMs) && completion.Task.Result;
        }
        catch { return false; }
    }

    public Dictionary<string, object?> Metrics()
    {
        var fault = Fault;
        return new Dictionary<string, object?>
        {
        ["records_written"] = Interlocked.Read(ref _recordsWritten),
        ["bytes_written"] = Interlocked.Read(ref _bytesWritten),
        ["queue_depth"] = _queue.Count,
        ["queue_high_watermark"] = Volatile.Read(ref _queueHighWatermark),
        ["serialize_ms"] = Math.Round(TimeSpan.FromTicks(Interlocked.Read(ref _serializeTicks)).TotalMilliseconds, 3),
        ["flush_ms"] = Math.Round(TimeSpan.FromTicks(Interlocked.Read(ref _flushTicks)).TotalMilliseconds, 3),
            ["fault"] = fault is null ? null : fault.GetType().Name + ": " + fault.Message
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            if (_queue.TryAdd(new StopItem(completion), 1000)) completion.Task.Wait(5000);
        }
        catch { }
        _queue.CompleteAdding();
        if (_thread.IsAlive) _thread.Join(1000);
        try { _writer.Dispose(); } catch { }
        try { _stream.Dispose(); } catch { }
        _queue.Dispose();
    }

    private void WriterLoop()
    {
        var recordsSinceFlush = 0;
        var lastFlush = Stopwatch.StartNew();
        try
        {
            foreach (var item in _queue.GetConsumingEnumerable())
            {
                switch (item)
                {
                    case RecordItem record:
                        record.Commit?.WaitForCompletion(10000);
                        WriteRecord(record.Record);
                        recordsSinceFlush++;
                        if (recordsSinceFlush >= 32 || lastFlush.ElapsedMilliseconds >= 1000)
                        {
                            TimedFlush(false);
                            recordsSinceFlush = 0;
                            lastFlush.Restart();
                        }
                        break;
                    case BarrierItem barrier:
                        TimedFlush(barrier.Durable);
                        recordsSinceFlush = 0;
                        lastFlush.Restart();
                        barrier.Completion.TrySetResult(true);
                        break;
                    case StopItem stop:
                        TimedFlush(true);
                        stop.Completion.TrySetResult(true);
                        return;
                }
            }
        }
        catch (Exception ex)
        {
            Volatile.Write(ref _fault, ex);
            while (_queue.TryTake(out var pending))
            {
                if (pending is BarrierItem barrier) barrier.Completion.TrySetException(ex);
                if (pending is StopItem stop) stop.Completion.TrySetException(ex);
            }
        }
    }

    private void WriteRecord(SortedDictionary<string, object?> record)
    {
        var started = Stopwatch.GetTimestamp();
        record["prev_record_sha256"] = _previousHash;
        var canonical = JsonSerializer.Serialize(record, JsonOptions);
        var hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
            Encoding.UTF8.GetBytes(canonical))).ToLowerInvariant();
        record["record_sha256"] = hash;
        var line = JsonSerializer.Serialize(record, JsonOptions);
        _writer.WriteLine(line);
        _previousHash = hash;
        Interlocked.Increment(ref _recordsWritten);
        Interlocked.Add(ref _bytesWritten, Encoding.UTF8.GetByteCount(line) + 1);
        Interlocked.Add(ref _serializeTicks, Stopwatch.GetElapsedTime(started).Ticks);
    }

    private void TimedFlush(bool durable)
    {
        var started = Stopwatch.GetTimestamp();
        _writer.Flush();
        if (durable) _stream.Flush(true);
        Interlocked.Add(ref _flushTicks, Stopwatch.GetElapsedTime(started).Ticks);
    }

    private void UpdateHighWatermark(int value)
    {
        while (true)
        {
            var current = Volatile.Read(ref _queueHighWatermark);
            if (value <= current || Interlocked.CompareExchange(ref _queueHighWatermark, value, current) == current) return;
        }
    }
}

internal sealed class ActionCommitToken
{
    private readonly ManualResetEventSlim _completed = new(false);
    private readonly Dictionary<string, object?> _payload;
    private int _committed;

    public ActionCommitToken(Dictionary<string, object?> payload) => _payload = payload;

    public void Complete(string status)
    {
        if (Interlocked.Exchange(ref _committed, 1) != 0) return;
        _payload["commit_status"] = status;
        _completed.Set();
    }

    public void WaitForCompletion(int timeoutMs) => _completed.Wait(timeoutMs);
}
