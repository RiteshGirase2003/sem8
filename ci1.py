# Define basic fuzzy set operations

def fuzzy_union(A, B):
    """Union of two fuzzy sets"""
    union_result = {}

    # Combine all elements from both sets
    all_elements = set(A.keys()).union(set(B.keys()))

    # For each element, take the maximum membership value
    for x in all_elements:
        a_value = A.get(x, 0)  # Membership value from set A (0 if not present)
        b_value = B.get(x, 0)  # Membership value from set B (0 if not present)
        union_result[x] = max(a_value, b_value)

    return union_result

def fuzzy_intersection(A, B):
    """Intersection of two fuzzy sets"""
    intersection_result = {}

    # Combine all elements from both sets
    all_elements = set(A.keys()).intersection(set(B.keys()))

    # For each element, take the minimum membership value
    for x in all_elements:
        a_value = A.get(x, 0)
        b_value = B.get(x, 0)
        intersection_result[x] = min(a_value, b_value)

    return intersection_result


def fuzzy_complement(A):
    """Complement of a fuzzy set"""
    complement_result = {}

    # For each element, subtract its membership value from 1
    for x in A:
        mu = A[x]  # Membership value
        complement_result[x] = 1 - mu

    return complement_result


def fuzzy_difference(A, B):
    """Difference of two fuzzy sets (A - B)"""
    difference_result = {}

    # Combine all elements from both sets
    all_elements = set(A.keys()).union(set(B.keys()))

    # For each element, do min(A(x), 1 - B(x))
    for x in all_elements:
        a_value = A.get(x, 0)
        b_value = B.get(x, 0)
        difference_result[x] = min(a_value, 1 - b_value)

    return difference_result


def cartesian_product(A, B):
    """Cartesian product of two fuzzy sets to create a fuzzy relation"""
    product_result = {}

    # For each pair (a, b), take the minimum membership value
    for a in A:
        for b in B:
            a_value = A[a]
            b_value = B[b]
            product_result[(a, b)] = min(a_value, b_value)

    return product_result


def max_min_composition(R1, R2):
    """Max-min composition of two fuzzy relations"""
    composition = {}
    # take keys of R1 and R2 put in set to get unique values only
    X = set(a for a, _ in R1)
    Z = set(c for _, c in R2)
    
    for x in X:
        for z in Z:
            # Find all y such that (x, y) in R1 and (y, z) in R2
            candidates = []
            for (a, b1), val1 in R1.items():
                for (b2, c), val2 in R2.items():
                    if a == x and b1 == b2 and c == z:
                        candidates.append(min(val1, val2))
            if candidates:
                composition[(x, z)] = max(candidates)
            else:
                composition[(x, z)] = 0
    return composition

# Example fuzzy sets
A = {'a': 0.2, 'b': 0.7, 'c': 1.0}
B = {'a': 0.5, 'b': 0.4, 'd': 0.8}

# Perform operations
print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

union = fuzzy_union(A, B)
print("\nUnion (A ∪ B):", union)

intersection = fuzzy_intersection(A, B)
print("\nIntersection (A ∩ B):", intersection)

complement_A = fuzzy_complement(A)
print("\nComplement (A'):", complement_A)

difference = fuzzy_difference(A, B)
print("\nDifference (A - B):", difference)

# Create fuzzy relations
relation_AB = cartesian_product(A, B)
relation_BA = cartesian_product(B, A)

print("\nFuzzy Relation (A × B):", relation_AB)
print("\nFuzzy Relation (B × A):", relation_BA)

# Max-min composition of relations
composition = max_min_composition(relation_AB, relation_BA)
print("\nMax-Min Composition (A × B) ∘ (B × A):", composition)
