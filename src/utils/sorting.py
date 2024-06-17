class MergeSort:

    @staticmethod
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        left_half = MergeSort.merge_sort(left_half)
        right_half = MergeSort.merge_sort(right_half)

        return MergeSort._merge(left_half, right_half)

    @staticmethod
    def _merge(left, right):
        result = []
        left_index, right_index = 0, 0
        left_subindex, right_subindex = 0, 0

        while (
            left_index < len(left)
            and right_index < len(right)
            and left_subindex < len(left[0])
            and right_subindex < len(right[0])
        ):
            if left[left_index][left_subindex] < right[right_index][right_subindex]:
                result.append(left[left_index])
                left_index += 1
                left_subindex = 0
            elif left[left_index][left_subindex] > right[right_index][right_subindex]:
                result.append(right[right_index])
                right_index += 1
                right_subindex += 0

            elif (
                left[left_index][left_subindex] == right[right_index][right_subindex]
                and left_subindex == (len(left[0]) - 1)
                and right_subindex == (len(right[0]) - 1)
            ):
                result.append(left[left_index])
                left_index += 1
                left_subindex = 0
                right_subindex = 0
            else:
                right_subindex += 1
                left_subindex += 1

        while left_index < len(left):
            result.append(left[left_index])
            left_index += 1

        while right_index < len(right):
            result.append(right[right_index])
            right_index += 1

        return result


if __name__ == "__main__":
    pixels = [
        [[1, 1, 3], [1, 1, 4], [0, 200, 5], [1, 1, 3], [0, 200, 5]],
        [[2, 2, 1], [0, 3, 2], [2, 3, 4], [0, 3, 1], [0, 3, 2]],
    ]
    sorted_pixels = [MergeSort.merge_sort(row) for row in pixels]
    print(sorted_pixels)
