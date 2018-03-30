import sys
import clang.cindex

function_calls = []             # List of AST node objects that are function calls
function_declarations = []      # List of AST node objects that are fucntion declarations

# Traverse the AST tree
def traverse(node):

    # Recurse for children of this node
    for child in node.get_children():
        traverse(child)

    # Add the node to function_calls
    if node.type == clang.cindex.CursorKind.CALL_EXPR:
        function_calls.append(node)

    # Add the node to function_declarations
    if node.type == clang.cindex.CursorKind.FUNCTION_DECL:
        function_declarations.append(node)

    # Print out information about the node
    print 'Found %s [line=%s, col=%s]' % (node.displayname, node.location.line, node.location.column)

# Tell clang.cindex where libclang.dylib is
index = clang.cindex.Index.create()

# Generate AST from filepath passed in the command line
tu = index.parse(sys.argv[1], args=["-x", "c++", "-std=c++11", "-D__global__=1"], options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
for i in tu.diagnostics:
    print(i)

root = tu.cursor        # Get the root of the AST
traverse(root)
