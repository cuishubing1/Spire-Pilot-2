using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Sts2HumanRecorder;

internal sealed record DecisionAnchor(
    long Sequence,
    int AttemptId,
    string ExactHash,
    string SemanticHash,
    string RoomKey,
    string Phase,
    int Round,
    int Turn);

internal sealed record RollbackMatch(
    DecisionAnchor? Anchor,
    string Quality,
    double Confidence,
    string Reason);

internal static class StateFingerprint
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
    private static readonly HashSet<string> VolatileKeys = new(StringComparer.Ordinal)
    {
        "instance_id", "lineage_quality", "engine_object_ref", "legal_actions", "capture_quality", "capture_errors", "display_name"
    };

    public static DecisionAnchor Build(Dictionary<string, object?> observation, long sequence, int attemptId)
    {
        using var document = JsonDocument.Parse(JsonSerializer.Serialize(observation, JsonOptions));
        var exact = HashNormalized(document.RootElement);
        var semantic = HashObject(SemanticProjection(observation));
        var run = Dict(observation.GetValueOrDefault("run"));
        var combat = Dict(observation.GetValueOrDefault("combat"));
        return new DecisionAnchor(
            sequence,
            attemptId,
            exact,
            semantic,
            RoomKey(run),
            observation.GetValueOrDefault("phase")?.ToString() ?? "unknown",
            Int(combat?.GetValueOrDefault("round")),
            Int(combat?.GetValueOrDefault("turn")));
    }

    public static DecisionAnchor Build(JsonElement observation, long sequence, int attemptId)
    {
        var value = JsonSerializer.Deserialize<Dictionary<string, object?>>(observation.GetRawText(), JsonOptions)
            ?? new Dictionary<string, object?>();
        return Build(value, sequence, attemptId);
    }

    public static RollbackMatch Match(DecisionAnchor current, IReadOnlyList<DecisionAnchor> history, int preferredAttempt)
    {
        var exact = history.LastOrDefault(x => x.AttemptId == preferredAttempt && x.ExactHash == current.ExactHash)
            ?? history.LastOrDefault(x => x.ExactHash == current.ExactHash);
        if (exact is not null)
            return new RollbackMatch(exact, "exact", 1.0, "normalized visible state hash matched");

        var semantic = history.LastOrDefault(x => x.AttemptId == preferredAttempt && x.SemanticHash == current.SemanticHash)
            ?? history.LastOrDefault(x => x.SemanticHash == current.SemanticHash);
        if (semantic is not null)
            return new RollbackMatch(semantic, "semantic", 0.9, "stable semantic state matched; volatile fields differed");

        if (!string.IsNullOrEmpty(current.RoomKey))
        {
            var roomStart = history.FirstOrDefault(x => x.AttemptId == preferredAttempt && x.RoomKey == current.RoomKey
                && x.Phase == current.Phase && (current.Phase != "combat_play" || (x.Round <= 1 && x.Turn <= 1)))
                ?? history.LastOrDefault(x => x.RoomKey == current.RoomKey && x.Phase == current.Phase
                    && (current.Phase != "combat_play" || (x.Round <= 1 && x.Turn <= 1)));
            if (roomStart is not null)
                return new RollbackMatch(roomStart, "room_entry", 0.7,
                    "same room and phase matched at the earliest observable combat turn");

            var location = history.LastOrDefault(x => x.AttemptId == preferredAttempt && x.RoomKey == current.RoomKey)
                ?? history.LastOrDefault(x => x.RoomKey == current.RoomKey);
            if (location is not null)
                return new RollbackMatch(location, "location_only", 0.5,
                    "only act/floor/room/map location matched; boundary must remain quarantined");
        }
        return new RollbackMatch(null, "unmatched", 0.0,
            "no prior visible state or room-entry anchor matched");
    }

    private static string HashNormalized(JsonElement element)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }))
            WriteNormalized(writer, element);
        return Convert.ToHexString(SHA256.HashData(stream.ToArray())).ToLowerInvariant();
    }

    private static void WriteNormalized(Utf8JsonWriter writer, JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject().Where(x => !VolatileKeys.Contains(x.Name)).OrderBy(x => x.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteNormalized(writer, property.Value);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray()) WriteNormalized(writer, item);
                writer.WriteEndArray();
                break;
            default:
                element.WriteTo(writer);
                break;
        }
    }

    private static object SemanticProjection(Dictionary<string, object?> observation)
    {
        var run = Dict(observation.GetValueOrDefault("run"));
        var player = Dict(observation.GetValueOrDefault("player"));
        var combat = Dict(observation.GetValueOrDefault("combat"));
        return new SortedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["phase"] = observation.GetValueOrDefault("phase")?.ToString(),
            ["room"] = RoomKey(run),
            ["player"] = new object?[]
            {
                player?.GetValueOrDefault("character_id"), player?.GetValueOrDefault("hp"),
                player?.GetValueOrDefault("max_hp"), player?.GetValueOrDefault("gold"),
                Models(player?.GetValueOrDefault("deck")), Models(player?.GetValueOrDefault("relics")),
                Models(player?.GetValueOrDefault("potions"))
            },
            ["combat"] = combat is null ? null : new object?[]
            {
                combat.GetValueOrDefault("round"), combat.GetValueOrDefault("turn"),
                combat.GetValueOrDefault("turn_phase"), combat.GetValueOrDefault("energy"),
                Models(combat.GetValueOrDefault("hand")), combat.GetValueOrDefault("draw_pile_count"),
                combat.GetValueOrDefault("discard_pile_count"), combat.GetValueOrDefault("exhaust_pile_count"),
                Creatures(combat.GetValueOrDefault("enemies"))
            },
            ["selection"] = Selection(observation)
        };
    }

    private static object? Selection(Dictionary<string, object?> observation)
    {
        var phase = observation.GetValueOrDefault("phase")?.ToString();
        var section = Dict(observation.GetValueOrDefault(phase ?? ""));
        if (phase is "card_reward" or "card_select") return Models(section?.GetValueOrDefault("cards"));
        if (phase == "bundle_select") return Items(section?.GetValueOrDefault("bundles")).Select(bundle =>
            Models(Dict(bundle)?.GetValueOrDefault("cards"))).ToList();
        if (phase == "reward_select") return Items(section?.GetValueOrDefault("rewards")).Select(reward =>
        {
            var row = Dict(reward);
            return new object?[] { row?.GetValueOrDefault("index"), row?.GetValueOrDefault("type"),
                row?.GetValueOrDefault("id"), row?.GetValueOrDefault("selected") };
        }).ToList();
        if (phase == "treasure") return Models(section?.GetValueOrDefault("relics"));
        if (phase == "relic_select") return Models(section?.GetValueOrDefault("relics"));
        return null;
    }

    private static object Models(object? value) => Items(value).Select(x =>
    {
        var row = Dict(x);
        var enchantment = Dict(row?.GetValueOrDefault("enchantment"));
        var affliction = Dict(row?.GetValueOrDefault("affliction"));
        return new object?[]
        {
            row?.GetValueOrDefault("id"), row?.GetValueOrDefault("upgrade_level"), row?.GetValueOrDefault("cost"),
            enchantment?.GetValueOrDefault("id"), enchantment?.GetValueOrDefault("amount"), enchantment?.GetValueOrDefault("status"),
            affliction?.GetValueOrDefault("id"), affliction?.GetValueOrDefault("amount"),
            row?.GetValueOrDefault("status"), row?.GetValueOrDefault("visible_state"),
            row?.GetValueOrDefault("persistent_state"), row?.GetValueOrDefault("runtime_state")
        };
    }).ToList();

    private static object Creatures(object? value) => Items(value).Select(x =>
    {
        var row = Dict(x);
        return new object?[]
        {
            row?.GetValueOrDefault("id"), row?.GetValueOrDefault("hp"), row?.GetValueOrDefault("max_hp"),
            row?.GetValueOrDefault("block"), row?.GetValueOrDefault("intends_attack"), row?.GetValueOrDefault("intent")
        };
    }).ToList();

    private static string RoomKey(Dictionary<string, object?>? run)
    {
        if (run is null) return "";
        var coord = Dict(run.GetValueOrDefault("map_coord"));
        return string.Join(":", new[]
        {
            run.GetValueOrDefault("act")?.ToString() ?? "",
            run.GetValueOrDefault("total_floor")?.ToString() ?? "",
            run.GetValueOrDefault("room_type")?.ToString() ?? "",
            coord?.GetValueOrDefault("col")?.ToString() ?? "",
            coord?.GetValueOrDefault("row")?.ToString() ?? ""
        });
    }

    private static string HashObject(object value) => Convert.ToHexString(SHA256.HashData(
        Encoding.UTF8.GetBytes(JsonSerializer.Serialize(value, JsonOptions)))).ToLowerInvariant();

    private static Dictionary<string, object?>? Dict(object? value)
    {
        if (value is Dictionary<string, object?> dictionary) return dictionary;
        if (value is JsonElement element && element.ValueKind == JsonValueKind.Object)
            return JsonSerializer.Deserialize<Dictionary<string, object?>>(element.GetRawText(), JsonOptions);
        return null;
    }

    private static IEnumerable<object?> Items(object? value)
    {
        if (value is IEnumerable<object?> items) return items;
        if (value is JsonElement element && element.ValueKind == JsonValueKind.Array)
            return element.EnumerateArray().Select(x => (object?)x).ToList();
        return Array.Empty<object?>();
    }

    private static int Int(object? value)
    {
        if (value is JsonElement element && element.TryGetInt32(out var parsed)) return parsed;
        try { return value is null ? 0 : Convert.ToInt32(value); } catch { return 0; }
    }
}
