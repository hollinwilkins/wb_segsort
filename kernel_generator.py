class KernelGenerator:
    _tmp_index: int

    def __init__(self):
        self._tmp_index = 0

    def tmp(self) -> str:
        tmp_index = self._tmp_index
        self._tmp_index += 1
        return f"tmp_{tmp_index}"

    def swap(self, a: int, b: int) -> str:
        tk = self.tmp()
        tv = self.tmp()
        return (f"{{ let {tk} = keys[{a}]; keys[{a}] = keys[{b}]; keys[{b}] = {tk};" +
                f"let {tv} = values[{a}]; values[{a}] = values[{b}]; values[{b}] = {tv}; }}")

    def cmp_swap(self, a: int, b: int) -> str:
        return (f"if keys[{a}] > keys[{b}] || (keys[{a}] == keys[{b}] && values[{a}] > values[{b}]) {{" +
                self.swap(a, b) + " }")

    def eql_swap(self, a: int, b: int) -> str:
        return (f"if keys[{a}] != keys[{b}] {{" +
                self.swap(a, b) + " }")

def main():
    generator = KernelGenerator()
    print("Swap:")
    print(generator.swap(0, 1))
    print("cmp_swap:")
    print(generator.cmp_swap(0, 1))
    print("eql_swap:")
    print(generator.eql_swap(0, 1))

if __name__ == "__main__":
    main()
