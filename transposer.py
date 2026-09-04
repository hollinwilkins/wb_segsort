class Transposer:
    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.WPT = N // M
        self.lane_swaps = (M.bit_length() - 1)
        self.register_swaps = (self.WPT.bit_length() - 1)

        # initial state after sorting is:
        # lane_bits | register_bits (lane bits hi, register bits lo)
        # target bit state after transpose is:
        # register bits (unsorted) : lane_bits (sorted) means (register bits hi, lane bits lo)
        # register bits do not need to be sorted because we will build a remap
        #   array that maps the register index to global index
        self.register_bits = [i for i in range(self.register_swaps)]
        self.lane_bits = [i + self.register_swaps for i in range(self.lane_swaps)]

    def swap(self, lane_bit: int, register_bit: int):
        self.lane_bits[lane_bit], self.register_bits[register_bit] = self.register_bits[register_bit], self.lane_bits[lane_bit]

    def build(self):
        # print(f"N: {self.N}, M: {self.M}, WPT: {self.WPT}, paral_swaps lane_swaps: {self.lane_swaps}, register_swaps: {self.register_swaps}")

        if self.lane_swaps == 0 or self.register_swaps == 0:
            return [], [0]

        # print(f"Before Swap: lane_bits: {self.lane_bits}, register_bits: {self.register_bits}")

        # generate swap pairs
        swaps = []

        for lane_bit in range(self.lane_swaps):
            register_bit = lane_bit % self.register_swaps
            self.swap(lane_bit, register_bit)
            swaps.append((lane_bit, register_bit))

        # print(f"After Swap: lane_bits: {self.lane_bits}, register_bits: {self.register_bits}")

        # print(f"swaps: {swaps}")

        # generate register indices
        bit_map = [bit - self.lane_swaps for bit in self.register_bits]
        register_map = []
        # print(f"bit map: {bit_map}")
        for ri in range(self.WPT):
            r = 0
            for j, bit in enumerate(bit_map):
                if ri & (1 << j):
                    r = r | (1 << bit)

            register_map.append(r)

        # print(f"register_map: {register_map}")
        # print()

        return swaps, register_map

def test_transpose(N: int, M: int):
    t = Transposer(N, M)

    # original block matrix
    # rows = lanes
    # cols = registers

    matrix = []
    for row in range(M):
        rv = []
        matrix.append(rv)
        for col in range(t.WPT):
            rv.append(row * t.WPT + col)

    # expected write output result
    # rows = lanes
    # cols = registers

    transpose_matrix = []
    for row in range(M):
        rv = []
        transpose_matrix.append(rv)
        for col in range(t.WPT):
            rv.append(col * M + row)

    swaps, register_map = t.build()

    for lane_bit, register_bit in swaps:
        for lane in range(M):
            # simulate lanes
            lane_pair = lane ^ pow(2, lane_bit)

            for r in range(t.WPT):
                # fine the right-side register
                if r & (1 << register_bit) == 0: continue # skip registers without bit set

                register_index = r # right-side for lo lane
                exchange_register_index = r ^ pow(2, register_bit) # left-side for hi lane

                if lane & pow(2, lane_bit) == 0: # check this is the low lane
                    matrix[lane][register_index], matrix[lane_pair][exchange_register_index] = matrix[lane_pair][exchange_register_index], matrix[lane][register_index]

    for row in matrix:
        row_copy = row.copy()
        for i in range(t.WPT):
            row[register_map[i]] = row_copy[i]

    # print("matrix:")
    # for row in matrix:
    #     print(f"{row}")

    # print("transpose_matrix:")
    # for row in transpose_matrix:
    #     print(f"{row}")

    return matrix == transpose_matrix

def build_swaps(N: int, M: int):
    transposer = Transposer(N, M)
    return transposer.build()

TESTS = [
    (4, 2),
    (8, 2),
    (16, 2),
    (4, 4),
    (8, 4),
    (16, 4),
    (16, 8),
    (32, 8),
    (32, 8),
    (64, 32),
    (128, 16),
    (128, 32),
    (128, 64),
]

def main():
    total = len(TESTS)
    failed = 0
    passed = 0

    for test in TESTS:
        N, M = test
        if not test_transpose(N, M):
            failed += 1
            print(f"N({N}), M({M}) FAILED!!!")
        else:
            passed += 1

    if failed > 0:
        print(f"failed: {failed}/{total}")
    print(f"passed: {passed}/{total}")

if __name__ == "__main__":
    main()

