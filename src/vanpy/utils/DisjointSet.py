# Taken from https://www.geeksforgeeks.org/disjoint-set-data-structures/
# implemented calculate_group_index

class DisjointSet:
    def __init__(self, n):
        # Constructor to create and
        # initialize sets of n items
        self.rank = [1] * n
        self.parent = list(range(n))
        self.group_index = [-1] * n

    # Finds set of given item x
    def find(self, x):
        # Finds the representative of the set
        # that x is an element of
        if (self.parent[x] != x):
            # if x is not the parent of itself
            # Then x is not the representative of
            # its set,
            self.parent[x] = self.find(self.parent[x])

            # so we recursively call Find on its parent
            # and move i's node directly under the
            # representative of this set

        return self.parent[x]

    # Do union of two sets represented
    # by x and y.
    def union(self, x, y):

        # Find current sets of x and y
        xset = self.find(x)
        yset = self.find(y)

        # If they are already in same set
        if xset == yset:
            return

        # Put smaller ranked item under
        # bigger ranked item if ranks are
        # different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset

        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset

        # If ranks are same, then move y under
        # x (doesn't matter which one goes where)
        # and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1

    def calculate_group_index(self):
        counter = 0
        seen = {}
        for i, p in enumerate(self.parent):
            if p not in seen:
                self.group_index[i] = counter
                seen[p] = counter
                counter += 1
            else:
                self.group_index[i] = seen[p]
        return self.group_index
