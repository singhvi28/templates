class BTrieNode:
    def __init__(self):
        self.children = [None, None]  # Index 0 -> bit 0, Index 1 -> bit 1
        self.is_end_of_number = False

class BinaryTrie:
    def __init__(self):
        self.root = BTrieNode()

    def insert(self, number):
        node = self.root
        for i in reversed(range(32)):  # assuming 32-bit integers
            bit = (number >> i) & 1
            if not node.children[bit]:
                node.children[bit] = BTrieNode()
            node = node.children[bit]
        node.is_end_of_number = True

    def search(self, number):
        node = self.root
        for i in reversed(range(32)):
            bit = (number >> i) & 1
            if not node.children[bit]:
                return False
            node = node.children[bit]
        return node.is_end_of_number

    def find_max_xor(self, number):
        """
        Finds the number in the trie that gives the maximum XOR with the given number.
        """
        node = self.root
        if not node.children[0] and not node.children[1]:
            return None  # Trie is empty

        max_xor = 0
        for i in reversed(range(32)):
            bit = (number >> i) & 1
            toggled_bit = 1 - bit
            if node.children[toggled_bit]:
                max_xor |= (1 << i)
                node = node.children[toggled_bit]
            else:
                node = node.children[bit]
        return max_xor


