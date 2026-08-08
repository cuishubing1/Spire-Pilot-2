using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: Sts2ContentAudit <game-data-dir> <output-json>");
    return 2;
}

var dataDir = Path.GetFullPath(args[0]);
var outputPath = Path.GetFullPath(args[1]);
AssemblyLoadContext.Default.Resolving += (_, name) =>
{
    var path = Path.Combine(dataDir, name.Name + ".dll");
    return File.Exists(path) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(path) : null;
};

var assemblyPath = Path.Combine(dataDir, "sts2.dll");
var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(assemblyPath);
var allTypes = assembly.GetTypes();
var families = new[]
{
    new Family("card", "MegaCrit.Sts2.Core.Models.CardModel", "MegaCrit.Sts2.Core.Models.Cards."),
    new Family("enchantment", "MegaCrit.Sts2.Core.Models.EnchantmentModel", "MegaCrit.Sts2.Core.Models.Enchantments."),
    new Family("affliction", "MegaCrit.Sts2.Core.Models.AfflictionModel", "MegaCrit.Sts2.Core.Models.Afflictions."),
    new Family("relic", "MegaCrit.Sts2.Core.Models.RelicModel", "MegaCrit.Sts2.Core.Models.Relics.")
};

var rows = new List<ContentRow>();
foreach (var family in families)
{
    var baseType = allTypes.Single(type => type.FullName == family.BaseType);
    foreach (var type in allTypes
        .Where(type => type.IsClass && !type.IsAbstract && type.FullName?.StartsWith(family.Namespace, StringComparison.Ordinal) == true)
        .Where(type => type != baseType && baseType.IsAssignableFrom(type))
        .Where(type => !type.Name.Contains('<'))
        .OrderBy(type => type.FullName, StringComparer.Ordinal))
    {
        var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;
        var properties = type.GetProperties(flags)
            .Where(property => property.GetIndexParameters().Length == 0)
            .Select(property => new MemberRow(
                property.Name,
                FriendlyName(property.PropertyType),
                property.GetMethod?.IsPublic == true,
                property.SetMethod is not null,
                IsSimple(property.PropertyType),
                property.CustomAttributes.Any(attribute => attribute.AttributeType.FullName == "MegaCrit.Sts2.Core.Saves.Runs.SavedPropertyAttribute")))
            .OrderBy(property => property.Name, StringComparer.Ordinal)
            .ToList();
        var fields = type.GetFields(flags)
            .Where(field => !field.IsStatic && !field.IsDefined(typeof(CompilerGeneratedAttribute), false))
            .Where(field => !typeof(Delegate).IsAssignableFrom(field.FieldType))
            .Select(field => new MemberRow(
                field.Name,
                FriendlyName(field.FieldType),
                field.IsPublic,
                !field.IsInitOnly,
                IsSimple(field.FieldType),
                field.CustomAttributes.Any(attribute => attribute.AttributeType.FullName == "MegaCrit.Sts2.Core.Saves.Runs.SavedPropertyAttribute")))
            .OrderBy(field => field.Name, StringComparer.Ordinal)
            .ToList();
        var methods = type.GetMethods(flags)
            .Where(method => !method.IsSpecialName && !method.Name.StartsWith('<'))
            .Where(method => method.GetBaseDefinition().DeclaringType != method.DeclaringType || LooksLikeHook(method.Name))
            .OrderBy(method => method.Name, StringComparer.Ordinal)
            .ToList();
        var hooks = methods.Select(method => method.Name).Distinct(StringComparer.Ordinal).ToList();
        var effects = methods.SelectMany(IlReferences)
            .Where(reference => reference.Contains("MegaCrit.Sts2.Core.Commands.", StringComparison.Ordinal)
                || reference.Contains("MegaCrit.Sts2.Core.Models.Powers.", StringComparison.Ordinal)
                || reference.Contains("MegaCrit.Sts2.Core.Models.Enchantments.", StringComparison.Ordinal)
                || reference.Contains("MegaCrit.Sts2.Core.Models.Afflictions.", StringComparison.Ordinal)
                || reference.Contains("MegaCrit.Sts2.Core.Models.Relics.", StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(reference => reference, StringComparer.Ordinal)
            .ToList();

        rows.Add(new ContentRow(
            family.Name,
            type.Name,
            type.FullName!,
            InferScope(hooks),
            properties,
            fields,
            hooks,
            effects));
    }
}

var report = new
{
    generated_at = DateTimeOffset.UtcNow,
    assembly = new
    {
        path = assemblyPath,
        version = assembly.GetName().Version?.ToString(),
        sha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(assemblyPath)))
    },
    counts = rows.GroupBy(row => row.Family).ToDictionary(group => group.Key, group => group.Count()),
    content = rows
};
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"wrote {rows.Count} rows to {outputPath}");
foreach (var group in rows.GroupBy(row => row.Family))
    Console.WriteLine($"{group.Key}: {group.Count()}, declared-state candidates: {group.Count(HasStateCandidate)}");
return 0;

static bool HasStateCandidate(ContentRow row) =>
    row.Properties.Any(member => member.Writable && member.Simple && !LooksConstant(member.Name))
    || row.Fields.Any(member => member.Writable && member.Simple && !LooksConstant(member.Name));

static bool LooksConstant(string name) =>
    name.Contains("Amount", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Count", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Damage", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Block", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Hp", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Gold", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Energy", StringComparison.OrdinalIgnoreCase)
    || name.Contains("Magic", StringComparison.OrdinalIgnoreCase);

static bool LooksLikeHook(string name) =>
    name.StartsWith("Before", StringComparison.Ordinal)
    || name.StartsWith("After", StringComparison.Ordinal)
    || name.StartsWith("On", StringComparison.Ordinal)
    || name.StartsWith("Try", StringComparison.Ordinal)
    || name.StartsWith("Modify", StringComparison.Ordinal)
    || name.StartsWith("Enchant", StringComparison.Ordinal)
    || name.StartsWith("Can", StringComparison.Ordinal)
    || name.StartsWith("Get", StringComparison.Ordinal);

static string InferScope(IReadOnlyCollection<string> hooks)
{
    var combat = hooks.Any(name => ContainsAny(name, "Combat", "Turn", "CardPlayed", "CardDrawn", "Attack", "Damage", "Block", "Power", "Enemy", "Orb", "Play"));
    var run = hooks.Any(name => ContainsAny(name, "Run", "Room", "Map", "Act", "Reward", "Shop", "Rest", "Treasure", "Gold", "Deck", "Potion", "CardAdded", "CardRemoved"));
    return (combat, run) switch
    {
        (true, true) => "combat_and_run",
        (true, false) => "combat",
        (false, true) => "run",
        _ => "static_or_on_acquire"
    };
}

static bool ContainsAny(string value, params string[] fragments) =>
    fragments.Any(fragment => value.Contains(fragment, StringComparison.OrdinalIgnoreCase));

static bool IsSimple(Type type)
{
    type = Nullable.GetUnderlyingType(type) ?? type;
    return type.IsPrimitive || type.IsEnum || type == typeof(string) || type == typeof(decimal);
}

static string FriendlyName(Type type)
{
    if (!type.IsGenericType) return type.FullName ?? type.Name;
    var root = (type.GetGenericTypeDefinition().FullName ?? type.Name).Split('`')[0];
    return root + "<" + string.Join(",", type.GetGenericArguments().Select(FriendlyName)) + ">";
}

static IEnumerable<string> IlReferences(MethodInfo method)
{
    var bytes = method.GetMethodBody()?.GetILAsByteArray();
    if (bytes is null) yield break;
    var oneByte = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(field => (OpCode)field.GetValue(null)!).Where(opcode => opcode.Size == 1).ToDictionary(opcode => (byte)opcode.Value);
    var twoByte = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(field => (OpCode)field.GetValue(null)!).Where(opcode => opcode.Size == 2).ToDictionary(opcode => (byte)(opcode.Value & 0xff));
    var module = method.Module;
    var typeArgs = method.DeclaringType?.IsGenericType == true ? method.DeclaringType.GetGenericArguments() : null;
    var methodArgs = method.IsGenericMethod ? method.GetGenericArguments() : null;
    for (var index = 0; index < bytes.Length;)
    {
        var first = bytes[index++];
        var opcode = first == 0xfe ? twoByte[bytes[index++]] : oneByte[first];
        var size = opcode.OperandType switch
        {
            OperandType.InlineNone => 0,
            OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
            OperandType.InlineVar => 2,
            OperandType.InlineI or OperandType.InlineBrTarget or OperandType.ShortInlineR
                or OperandType.InlineField or OperandType.InlineMethod or OperandType.InlineSig
                or OperandType.InlineString or OperandType.InlineTok or OperandType.InlineType => 4,
            OperandType.InlineI8 or OperandType.InlineR => 8,
            OperandType.InlineSwitch => 4 + BitConverter.ToInt32(bytes, index) * 4,
            _ => 0
        };
        if (size == 4 && opcode.OperandType is OperandType.InlineField or OperandType.InlineMethod or OperandType.InlineTok or OperandType.InlineType)
        {
            var token = BitConverter.ToInt32(bytes, index);
            string? resolved = null;
            try
            {
                var member = module.ResolveMember(token, typeArgs, methodArgs);
                if (member is not null) resolved = member.ToString() ?? member.Name;
            }
            catch { }
            if (resolved is not null) yield return resolved;
        }
        index += size;
    }
}

internal sealed record Family(string Name, string BaseType, string Namespace);
internal sealed record MemberRow(string Name, string Type, bool Public, bool Writable, bool Simple, bool Saved);
internal sealed record ContentRow(
    string Family,
    string TypeName,
    string FullName,
    string Scope,
    List<MemberRow> Properties,
    List<MemberRow> Fields,
    List<string> Hooks,
    List<string> EffectReferences);
