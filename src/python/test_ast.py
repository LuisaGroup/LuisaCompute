import inspect
import ast
import astpretty
import astor
import taichi as ti


class IfScope:
    def __init__(self, stmt, is_true):
        self._stmt = stmt
        self._is_true = is_true

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class IfStmt:
    def __init__(self, cond):
        self._cond = cond

    @property
    def condition(self):
        return self._cond

    def scope(self, branch):
        return IfScope(self, branch)


class IfStmtTransformer(ast.NodeTransformer):

    def __init__(self):
        self._insertions = []
        self._cond_counter = 0

    @property
    def insertions(self):
        return self._insertions

    def visit_If(self, node: ast.If):
        cond_name = f"cond_{self._cond_counter}"
        self._cond_counter += 1
        assign = ast.Assign(
            targets=[ast.Name(id=cond_name, ctx=ast.Store())],
            value=node.test)
        self._insertions.append((node, assign))
        node.test = ast.Name(id=cond_name, ctx=ast.Load())
        for i, child in enumerate(node.body):
            node.body[i] = self.visit(child)
        for i, child in enumerate(node.orelse):
            child.from_body = False
            node.orelse[i] = self.visit(child)
        return node


def f(a: int, b: int):
    if a < b:
        x = a
    else:
        if a == b:
            x = b
        else:
            x = b

    stmt = IfStmt(a == b)
    with stmt.scope(True):
        pass
    with stmt.scope(False):
        pass
    return x


if __name__ == "__main__":
    src = inspect.getsource(f)
    tree = ast.parse(src, type_comments=True)
    astpretty.pprint(tree)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            child.from_body = True
    transformer = IfStmtTransformer()
    transformer.visit(tree)
    ast.fix_missing_locations(tree)
    for if_stmt, assign_stmt in transformer.insertions:
        if if_stmt.from_body:
            scope: list = if_stmt.parent.body
        else:
            scope: list = if_stmt.parent.orelse
        scope.insert(scope.index(if_stmt), assign_stmt)
    ast.fix_missing_locations(tree)
    src = astor.to_source(tree)
    print(src)

ti.init()

res = (510, 510)
pixels = ti.Vector.field(3, dtype=float, shape=res)


def good(a):
    print("Hello", a)


@ti.kernel
def paint(w: int, h: int):
    for i, j in ti.ndrange(w, h):
        u = i / w
        v = j / h
        pixels[i, j] = [u, v, 0]


gui = ti.GUI('UV', res)
while not gui.get_event(ti.GUI.ESCAPE):
    ti.impl.get_runtime().print_preprocessed = True
    paint(480, 480)
    gui.set_image(pixels)
    gui.show()
