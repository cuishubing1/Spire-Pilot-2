using System.Collections;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;

namespace Sts2HumanRecorder;

internal static class ContentProvenance
{
    private static readonly Assembly GameAssembly = typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly;
    private static readonly Assembly RecorderAssembly = typeof(ContentProvenance).Assembly;
    private static readonly Dictionary<Assembly, Dictionary<string, object?>> IdentityCache = new();
    private static readonly object EnvironmentGate = new();
    private static Dictionary<string, object?>? _environmentCache;
    private static string? _environmentAssemblySignature;

    public static void PrewarmEnvironmentSummary()
    {
        _ = Task.Run(async () =>
        {
            // Let the mod loader finish registering the remaining assemblies,
            // then move DLL hashing/type inspection away from the first climb.
            await Task.Delay(750).ConfigureAwait(false);
            try { _ = EnvironmentSummary(); } catch { }
        });
    }

    public static void AddEntitySource(Dictionary<string, object?> row, object? value)
    {
        var model = UnwrapModel(value);
        var assembly = model?.GetType().Assembly;
        row["source_kind"] = SourceKind(assembly);
        row["source_assembly"] = assembly?.GetName().Name;
        row["source_mod_id"] = assembly is null ? null : CachedModIdentity(assembly).GetValueOrDefault("id");
    }

    public static Dictionary<string, object?> EnvironmentSummary()
    {
        var signature = CurrentModAssemblySignature();
        lock (EnvironmentGate)
            if (_environmentCache is not null && _environmentAssemblySignature == signature) return _environmentCache;
        var mods = LoadedMods();
        var summary = new Dictionary<string, object?>
        {
            ["loaded_mods"] = mods,
            ["has_content_mods"] = mods.Any(row => ReflectionUtil.Bool(row.GetValueOrDefault("defines_game_entities"))
                || ReflectionUtil.Bool(row.GetValueOrDefault("declared_affects_gameplay"))),
            ["provenance_method"] = "loaded_assembly_and_entity_type_v1"
        };
        lock (EnvironmentGate)
        {
            _environmentCache = summary;
            _environmentAssemblySignature = signature;
            return _environmentCache;
        }
    }

    private static string CurrentModAssemblySignature() => string.Join("|",
        AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => assembly != RecorderAssembly && IsModAssembly(assembly))
            .Select(assembly => assembly.GetName().Name + "@" + assembly.GetName().Version)
            .OrderBy(value => value, StringComparer.OrdinalIgnoreCase));

    public static string ClassifyDecision(Dictionary<string, object?> observation, Dictionary<string, object?> action)
    {
        if (!string.Equals(observation.GetValueOrDefault("capture_quality")?.ToString(), "complete", StringComparison.Ordinal))
            return "unknown";
        var kinds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        CollectSourceKinds(observation, kinds);
        CollectSourceKinds(action, kinds);
        if (kinds.Contains("mod")) return "modded";
        if (kinds.Contains("unknown") || kinds.Contains("external")) return "unknown";
        return kinds.Contains("base_game") ? "base_game" : "unknown";
    }

    private static List<Dictionary<string, object?>> LoadedMods()
    {
        var result = new List<Dictionary<string, object?>>();
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies().OrderBy(x => x.GetName().Name, StringComparer.OrdinalIgnoreCase))
        {
            if (assembly == RecorderAssembly || !IsModAssembly(assembly)) continue;
            var identity = CachedModIdentity(assembly);
            result.Add(new Dictionary<string, object?>
            {
                ["id"] = identity.GetValueOrDefault("id"),
                ["name"] = identity.GetValueOrDefault("name"),
                ["version"] = identity.GetValueOrDefault("version") ?? assembly.GetName().Version?.ToString(),
                ["workshop_id"] = WorkshopId(assembly),
                ["assembly"] = assembly.GetName().Name,
                ["assembly_version"] = assembly.GetName().Version?.ToString(),
                ["assembly_sha256"] = AssemblySha256(assembly),
                ["declared_affects_gameplay"] = identity.GetValueOrDefault("affects_gameplay"),
                ["defines_game_entities"] = DefinesGameEntities(assembly)
            });
        }
        return result;
    }

    private static object? UnwrapModel(object? value)
    {
        if (value is null) return null;
        var inner = ReflectionUtil.Get(value, "Model", "CardModel", "RelicModel", "PotionModel", "Monster", "Character");
        return inner is null || inner is string || inner.GetType().IsPrimitive ? value : inner;
    }

    private static string SourceKind(Assembly? assembly)
    {
        if (assembly is null) return "unknown";
        if (assembly == GameAssembly) return "base_game";
        if (assembly == RecorderAssembly) return "recorder";
        return IsModAssembly(assembly) ? "mod" : "external";
    }

    private static bool IsModAssembly(Assembly assembly)
    {
        var location = SafeLocation(assembly).Replace('\\', '/');
        return location.Contains("/mods/", StringComparison.OrdinalIgnoreCase)
            || location.Contains("/workshop/content/2868840/", StringComparison.OrdinalIgnoreCase);
    }

    private static string SafeLocation(Assembly assembly)
    {
        try { return assembly.Location ?? string.Empty; }
        catch { return string.Empty; }
    }

    private static Dictionary<string, object?> ModIdentity(Assembly assembly)
    {
        var fallback = new Dictionary<string, object?> { ["id"] = assembly.GetName().Name };
        try
        {
            var directory = Path.GetDirectoryName(SafeLocation(assembly));
            if (string.IsNullOrWhiteSpace(directory)) return fallback;
            JsonElement root = default;
            JsonDocument? document = null;
            foreach (var candidate in Directory.EnumerateFiles(directory, "*.json", SearchOption.TopDirectoryOnly)
                         .Where(path => !path.EndsWith(".deps.json", StringComparison.OrdinalIgnoreCase)))
            {
                try
                {
                    var parsed = JsonDocument.Parse(File.ReadAllText(candidate));
                    if (parsed.RootElement.ValueKind == JsonValueKind.Object
                        && parsed.RootElement.TryGetProperty("id", out var id)
                        && id.ValueKind == JsonValueKind.String)
                    {
                        document = parsed;
                        root = document.RootElement;
                        break;
                    }
                    parsed.Dispose();
                }
                catch { }
            }
            if (document is null) return fallback;
            using (document)
            {
            object? Read(string name) => root.TryGetProperty(name, out var value) ? value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.String => value.GetString(),
                _ => value.ToString()
            } : null;
                return new Dictionary<string, object?>
                {
                    ["id"] = Read("id") ?? assembly.GetName().Name,
                    ["name"] = Read("name"),
                    ["version"] = Read("version"),
                    ["affects_gameplay"] = Read("affects_gameplay")
                };
            }
        }
        catch { return fallback; }
    }

    private static Dictionary<string, object?> CachedModIdentity(Assembly assembly)
    {
        lock (IdentityCache)
        {
            if (!IdentityCache.TryGetValue(assembly, out var identity))
            {
                identity = ModIdentity(assembly);
                IdentityCache[assembly] = identity;
            }
            return identity;
        }
    }

    private static string? WorkshopId(Assembly assembly)
    {
        var parts = SafeLocation(assembly).Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i + 1 < parts.Length; i++)
            if (parts[i] == "2868840") return parts[i + 1];
        return null;
    }

    private static string? AssemblySha256(Assembly assembly)
    {
        try
        {
            var location = SafeLocation(assembly);
            return string.IsNullOrWhiteSpace(location) ? null
                : Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(location))).ToLowerInvariant();
        }
        catch { return null; }
    }

    private static bool DefinesGameEntities(Assembly assembly)
    {
        try
        {
            return GetLoadableTypes(assembly).Any(type => BaseTypeNames(type).Any(name => name is
                "CardModel" or "RelicModel" or "PotionModel" or "CharacterModel" or "MonsterModel"));
        }
        catch { return false; }
    }

    private static IEnumerable<Type> GetLoadableTypes(Assembly assembly)
    {
        try { return assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { return ex.Types.Where(x => x is not null).Cast<Type>(); }
    }

    private static IEnumerable<string> BaseTypeNames(Type type)
    {
        for (var current = type; current is not null; current = current.BaseType)
            yield return current.Name;
    }

    private static void CollectSourceKinds(object? value, HashSet<string> kinds)
    {
        if (value is null || value is string) return;
        if (value is IDictionary dictionary)
        {
            foreach (DictionaryEntry entry in dictionary)
            {
                if (string.Equals(entry.Key?.ToString(), "source_kind", StringComparison.OrdinalIgnoreCase)
                    && entry.Value is not null) kinds.Add(entry.Value.ToString()!);
                else CollectSourceKinds(entry.Value, kinds);
            }
            return;
        }
        if (value is IEnumerable enumerable)
            foreach (var item in enumerable) CollectSourceKinds(item, kinds);
    }
}
