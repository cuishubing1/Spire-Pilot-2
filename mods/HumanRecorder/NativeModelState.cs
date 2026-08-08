using System.Collections;
using System.Text;

namespace Sts2HumanRecorder;

internal sealed record PersistentStateCapture(List<Dictionary<string, object?>> State, string Quality, string? Error);

internal static class NativeModelState
{
    public const string SchemaVersion = "native-model-state-0.1.0";

    public static List<Dictionary<string, object?>> PersistentState(object model, bool audit)
        => CapturePersistentState(model, audit).State;

    public static PersistentStateCapture CapturePersistentState(object model, bool audit)
    {
        try
        {
            var gameAssembly = typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly;
            var type = gameAssembly.GetType("MegaCrit.Sts2.Core.Saves.Runs.SavedProperties", true)!;
            var method = type.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                .Single(candidate => candidate.Name == "From" && candidate.GetParameters().Length == 1);
            var properties = method.Invoke(null, new[] { model });
            var sourceKind = model.GetType().Assembly == typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly
                ? "base_game" : "mod";
            var state = EncodeSavedProperties(properties, model.GetType().Name,
                audit || sourceKind == "base_game" ? (sourceKind == "base_game" ? "public" : "audit_only") : "omit");
            return new PersistentStateCapture(state, "complete", null);
        }
        catch (Exception ex)
        {
            if (!IsNativeStateModel(model)) return new PersistentStateCapture(new(), "not_applicable", null);
            var error = (ex.InnerException ?? ex).GetType().Name + ": " + (ex.InnerException ?? ex).Message;
            return new PersistentStateCapture(new(), "partial", error);
        }
    }

    public static List<Dictionary<string, object?>> RuntimeState(object model, IReadOnlyList<StateMember> members)
    {
        var result = new List<Dictionary<string, object?>>();
        foreach (var member in members)
        {
            var value = ReflectionUtil.Get(model, member.Member);
            if (value is null) continue;
            var row = EncodeValue(member.Key, value, member.Lifecycle, "public", "v0.107.1_adapter");
            if (row is not null) result.Add(row);
        }
        return result;
    }

    public static Dictionary<string, object?> Enchantment(object? enchantment, bool audit = false)
    {
        if (enchantment is null) return new();
        var persistent = CapturePersistentState(enchantment, audit);
        var result = new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(enchantment),
            ["amount"] = ReflectionUtil.Int(ReflectionUtil.Get(enchantment, "Amount")),
            ["status"] = ReflectionUtil.Get(enchantment, "Status")?.ToString(),
            ["show_amount"] = ReflectionUtil.Bool(ReflectionUtil.Get(enchantment, "ShowAmount")),
            ["display_amount"] = ReflectionUtil.Int(ReflectionUtil.Get(enchantment, "DisplayAmount")),
            ["dynamic_vars"] = DynamicValues(enchantment),
            ["persistent_state"] = persistent.State,
            ["persistent_state_capture_quality"] = persistent.Quality,
            ["persistent_state_capture_error"] = persistent.Error,
            ["runtime_state"] = RuntimeState(enchantment, ContentStateRegistry.EnchantmentRuntime(enchantment)),
            ["state_schema"] = SchemaVersion,
            ["projection_version"] = ContentStateRegistry.Version
        };
        ContentProvenance.AddEntitySource(result, enchantment);
        return result;
    }

    public static Dictionary<string, object?> Affliction(object? affliction)
    {
        if (affliction is null) return new();
        var result = new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(affliction),
            ["amount"] = ReflectionUtil.Int(ReflectionUtil.Get(affliction, "Amount")),
            ["state_schema"] = SchemaVersion
        };
        ContentProvenance.AddEntitySource(result, affliction);
        return result;
    }

    public static Dictionary<string, object?> Relic(object relic, bool audit = false)
    {
        var visible = new Dictionary<string, object?>
        {
            ["show_counter"] = ReflectionUtil.Bool(ReflectionUtil.Get(relic, "ShowCounter")),
            ["display_amount"] = ReflectionUtil.Int(ReflectionUtil.Get(relic, "DisplayAmount")),
            ["is_used_up"] = ReflectionUtil.Bool(ReflectionUtil.Get(relic, "IsUsedUp")),
            ["is_wax"] = ReflectionUtil.Bool(ReflectionUtil.Get(relic, "IsWax")),
            ["is_melted"] = ReflectionUtil.Bool(ReflectionUtil.Get(relic, "IsMelted"))
        };
        var persistent = CapturePersistentState(relic, audit);
        var result = new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(relic),
            ["stack_count"] = ReflectionUtil.Int(ReflectionUtil.Get(relic, "StackCount"), 1),
            ["floor_added"] = ReflectionUtil.Int(ReflectionUtil.Get(relic, "FloorAddedToDeck")),
            ["status"] = ReflectionUtil.Get(relic, "Status")?.ToString(),
            ["visible_state"] = visible,
            ["dynamic_vars"] = DynamicValues(relic),
            ["persistent_state"] = persistent.State,
            ["persistent_state_capture_quality"] = persistent.Quality,
            ["persistent_state_capture_error"] = persistent.Error,
            ["runtime_state"] = RuntimeState(relic, ContentStateRegistry.RelicRuntime(relic)),
            ["state_schema"] = SchemaVersion,
            ["projection_version"] = ContentStateRegistry.Version,
            ["projection_known"] = ContentStateRegistry.IsKnownBaseRelic(relic)
        };
        ContentProvenance.AddEntitySource(result, relic);
        return result;
    }

    public static Dictionary<string, object?> AuditCard(object card)
    {
        var energyCost = ReflectionUtil.Get(card, "EnergyCost");
        var persistent = CapturePersistentState(card, true);
        return new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(card),
            ["upgrade_level"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentUpgradeLevel")),
            ["floor_added"] = ReflectionUtil.Get(card, "FloorAddedToDeck"),
            ["energy_cost"] = new Dictionary<string, object?>
            {
                ["canonical"] = ReflectionUtil.Int(ReflectionUtil.Get(energyCost, "Canonical"), ReflectionUtil.Int(ReflectionUtil.Get(card, "CanonicalEnergyCost"))),
                ["current"] = ReflectionUtil.Int(ReflectionUtil.Call(energyCost, "GetResolved")),
                ["costs_x"] = ReflectionUtil.Bool(ReflectionUtil.Get(energyCost, "CostsX")),
                ["captured_x"] = ReflectionUtil.Get(energyCost, "CapturedXValue")
            },
            ["enchantment"] = NullIfEmpty(Enchantment(ReflectionUtil.Get(card, "Enchantment"), true)),
            ["affliction"] = NullIfEmpty(Affliction(ReflectionUtil.Get(card, "Affliction"))),
            ["persistent_state"] = persistent.State,
            ["persistent_state_capture_quality"] = persistent.Quality,
            ["persistent_state_capture_error"] = persistent.Error,
            ["runtime_state"] = RuntimeState(card, ContentStateRegistry.CardRuntime(card)),
            ["runtime_flags"] = new Dictionary<string, object?>
            {
                ["exhaust_on_next_play"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "ExhaustOnNextPlay")),
                ["retain_this_turn"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "ShouldRetainThisTurn")),
                ["sly_this_turn"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "IsSlyThisTurn"))
            }
        };
    }

    public static List<Dictionary<string, object?>> EncodeSavedProperties(object? properties, string modelType, string visibility)
    {
        var result = new List<Dictionary<string, object?>>();
        if (properties is null || visibility == "omit") return result;
        foreach (var group in new[] { "ints", "bools", "strings", "intArrays", "modelIds", "cards", "cardArrays" })
        {
            foreach (var item in ReflectionUtil.Items(ReflectionUtil.Get(properties, group)))
            {
                var name = ReflectionUtil.Get(item, "name", "Name")?.ToString();
                var value = ReflectionUtil.Get(item, "value", "Value");
                if (string.IsNullOrWhiteSpace(name) || value is null) continue;
                var row = EncodeValue(name, value, ContentStateRegistry.PersistentLifecycle(modelType, name), visibility, "saved_property");
                if (row is not null) result.Add(row);
            }
        }
        return result.OrderBy(row => row["key"]?.ToString(), StringComparer.Ordinal).ToList();
    }

    private static Dictionary<string, object?>? EncodeValue(string key, object value, string lifecycle, string visibility, string source)
    {
        var row = new Dictionary<string, object?>
        {
            ["key"] = key,
            ["normalized_key"] = SnakeCase(key),
            ["lifecycle"] = lifecycle,
            ["visibility"] = visibility,
            ["source"] = source
        };
        var type = Nullable.GetUnderlyingType(value.GetType()) ?? value.GetType();
        if (type == typeof(bool)) { row["value_type"] = "bool"; row["bool_value"] = value; }
        else if (type == typeof(byte) || type == typeof(sbyte) || type == typeof(short) || type == typeof(ushort)
            || type == typeof(int) || type == typeof(uint) || type == typeof(long) || type == typeof(ulong))
        { row["value_type"] = "int"; row["int_value"] = Convert.ToInt64(value); }
        else if (type == typeof(float) || type == typeof(double) || type == typeof(decimal))
        { row["value_type"] = "decimal"; row["decimal_value"] = Convert.ToDecimal(value); }
        else if (type.IsEnum) { row["value_type"] = "enum"; row["enum_value"] = value.ToString(); }
        else if (type == typeof(string)) { row["value_type"] = "string"; row["string_value"] = value.ToString(); }
        else if (value is int[] ints) { row["value_type"] = "int_list"; row["int_list_value"] = ints; }
        else if (IsSerializableCard(value)) { row["value_type"] = "card"; row["card_value"] = SerializableCard(value); }
        else if (value is IEnumerable enumerable && value is not string)
        {
            var cards = enumerable.Cast<object?>().Where(item => item is not null).Select(item => SerializableCard(item!)).ToList();
            row["value_type"] = "card_list"; row["card_list_value"] = cards;
        }
        else
        {
            var id = ReflectionUtil.Id(value);
            if (string.IsNullOrWhiteSpace(id)) return null;
            row["value_type"] = "model_id"; row["model_id_value"] = id;
        }
        return row;
    }

    private static Dictionary<string, object?> SerializableCard(object card)
    {
        return new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(card),
            ["upgrade_level"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentUpgradeLevel")),
            ["floor_added"] = ReflectionUtil.Get(card, "FloorAddedToDeck"),
            ["enchantment"] = SerializableEnchantment(ReflectionUtil.Get(card, "Enchantment")),
            ["persistent_state"] = EncodeSavedProperties(ReflectionUtil.Get(card, "Props"), card.GetType().Name, "audit_only")
        };
    }

    private static object? SerializableEnchantment(object? enchantment)
    {
        if (enchantment is null) return null;
        return new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(enchantment),
            ["amount"] = ReflectionUtil.Int(ReflectionUtil.Get(enchantment, "Amount")),
            ["persistent_state"] = EncodeSavedProperties(ReflectionUtil.Get(enchantment, "Props"), enchantment.GetType().Name, "audit_only")
        };
    }

    private static bool IsSerializableCard(object value) =>
        value.GetType().Name == "SerializableCard" || ReflectionUtil.Get(value, "CurrentUpgradeLevel") is not null;

    private static bool IsNativeStateModel(object model)
    {
        for (var type = model.GetType(); type is not null; type = type.BaseType)
            if (type.FullName is "MegaCrit.Sts2.Core.Models.CardModel" or "MegaCrit.Sts2.Core.Models.RelicModel"
                or "MegaCrit.Sts2.Core.Models.EnchantmentModel") return true;
        return false;
    }

    private static Dictionary<string, object?> DynamicValues(object model)
    {
        var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        var vars = ReflectionUtil.Get(model, "DynamicVars");
        foreach (var value in ReflectionUtil.Items(ReflectionUtil.Get(vars, "Values")))
        {
            var name = ReflectionUtil.Get(value, "Name")?.ToString()?.ToLowerInvariant();
            if (!string.IsNullOrWhiteSpace(name)) values[name] = ReflectionUtil.Int(ReflectionUtil.Get(value, "BaseValue"));
        }
        return values;
    }

    private static object? NullIfEmpty(Dictionary<string, object?> value) => value.Count == 0 ? null : value;

    private static string SnakeCase(string value)
    {
        var builder = new StringBuilder(value.Length + 8);
        for (var index = 0; index < value.Length; index++)
        {
            var current = value[index];
            if (char.IsUpper(current) && index > 0 && (char.IsLower(value[index - 1])
                || (index + 1 < value.Length && char.IsLower(value[index + 1])))) builder.Append('_');
            builder.Append(char.ToLowerInvariant(current));
        }
        return builder.ToString();
    }
}
