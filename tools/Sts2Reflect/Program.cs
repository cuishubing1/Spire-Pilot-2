using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Loader;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: Sts2Reflect <game-data-dir> <type-name-fragment> [member-fragment]");
    return 2;
}

var dataDir = Path.GetFullPath(args[0]);
AssemblyLoadContext.Default.Resolving += (_, name) =>
{
    var path = Path.Combine(dataDir, name.Name + ".dll");
    return File.Exists(path) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(path) : null;
};
var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(dataDir, "sts2.dll"));
var typeFilter = args[1];
var memberFilter = args.Length > 2 ? args[2] : "";
var dumpCalls = args.Contains("--calls", StringComparer.OrdinalIgnoreCase);
foreach (var type in assembly.GetTypes().Where(t => t.FullName?.Contains(typeFilter, StringComparison.OrdinalIgnoreCase) == true))
{
    Console.WriteLine($"TYPE {type.FullName}");
    var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
    foreach (var constructor in type.GetConstructors(flags))
        if (constructor.ToString()!.Contains(memberFilter, StringComparison.OrdinalIgnoreCase)) Console.WriteLine($"  CTOR {constructor}");
    foreach (var property in type.GetProperties(flags))
        if (property.Name.Contains(memberFilter, StringComparison.OrdinalIgnoreCase)) Console.WriteLine($"  PROP {property.PropertyType} {property.Name}");
    foreach (var field in type.GetFields(flags))
        if (field.Name.Contains(memberFilter, StringComparison.OrdinalIgnoreCase)) Console.WriteLine($"  FIELD {field.FieldType} {field.Name}");
    foreach (var method in type.GetMethods(flags))
        if (method.Name.Contains(memberFilter, StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"  METHOD {method}");
            if (dumpCalls)
                foreach (var reference in IlReferences(method)) Console.WriteLine($"    IL {reference}");
        }
}
return 0;

static IEnumerable<string> IlReferences(MethodInfo method)
{
    var bytes = method.GetMethodBody()?.GetILAsByteArray();
    if (bytes is null) yield break;
    var oneByte = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(x => (OpCode)x.GetValue(null)!).Where(x => x.Size == 1).ToDictionary(x => (byte)x.Value);
    var twoByte = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(x => (OpCode)x.GetValue(null)!).Where(x => x.Size == 2).ToDictionary(x => (byte)(x.Value & 0xff));
    var module = method.Module;
    var typeArgs = method.DeclaringType?.IsGenericType == true ? method.DeclaringType.GetGenericArguments() : null;
    var methodArgs = method.IsGenericMethod ? method.GetGenericArguments() : null;
    for (var i = 0; i < bytes.Length;)
    {
        var offset = i;
        var first = bytes[i++];
        var opcode = first == 0xfe ? twoByte[bytes[i++]] : oneByte[first];
        var size = opcode.OperandType switch
        {
            OperandType.InlineNone => 0,
            OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
            OperandType.InlineVar => 2,
            OperandType.InlineI or OperandType.InlineBrTarget or OperandType.ShortInlineR
                or OperandType.InlineField or OperandType.InlineMethod or OperandType.InlineSig
                or OperandType.InlineString or OperandType.InlineTok or OperandType.InlineType => 4,
            OperandType.InlineI8 or OperandType.InlineR => 8,
            OperandType.InlineSwitch => 4 + BitConverter.ToInt32(bytes, i) * 4,
            _ => 0
        };
        if (size == 4 && opcode.OperandType is OperandType.InlineField or OperandType.InlineMethod
            or OperandType.InlineString or OperandType.InlineTok or OperandType.InlineType)
        {
            var token = BitConverter.ToInt32(bytes, i);
            object? value = null;
            try
            {
                value = opcode.OperandType == OperandType.InlineString
                    ? module.ResolveString(token)
                    : module.ResolveMember(token, typeArgs, methodArgs);
            }
            catch { }
            var display = value is MemberInfo member
                ? $"{member.DeclaringType?.FullName}::{member}"
                : value?.ToString() ?? $"token 0x{token:x8}";
            yield return $"IL_{offset:x4}: {opcode.Name} {display}";
        }
        else if (opcode == OpCodes.Ldc_I4_M1 || opcode == OpCodes.Ldc_I4_0 || opcode == OpCodes.Ldc_I4_1
            || opcode == OpCodes.Ldc_I4_2 || opcode == OpCodes.Ldc_I4_3 || opcode == OpCodes.Ldc_I4_4
            || opcode == OpCodes.Ldc_I4_5 || opcode == OpCodes.Ldc_I4_6 || opcode == OpCodes.Ldc_I4_7
            || opcode == OpCodes.Ldc_I4_8)
            yield return $"IL_{offset:x4}: {opcode.Name}";
        else if (opcode == OpCodes.Ldc_I4_S)
            yield return $"IL_{offset:x4}: {opcode.Name} {unchecked((sbyte)bytes[i])}";
        else if (opcode == OpCodes.Ldc_I4)
            yield return $"IL_{offset:x4}: {opcode.Name} {BitConverter.ToInt32(bytes, i)}";
        i += size;
    }
}
