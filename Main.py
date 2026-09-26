def read_input(file_path):
    """Считывание матрицы A и вектора b из текстового файла."""
    a = []
    b = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = [float(x) for x in line_str.split()]
            a.append(parts[:-1])
            b.append(parts[-1])
    return a, b


def solve_seidel(a, b, eps=1e-6, max_iterations=1000):
    """Решение СЛАУ методом Зейделя.

    :param a: матрица коэффициентов A
    :param b: вектор свободных членов b
    :param eps: точность остановки итераций
    :param max_iterations: максимальное количество итераций
    """
    n = len(b)
    x = [0.0] * n  # Начальное приближение x^(0) = 0

    # Проверка наличия нулей на главной диагонали
    for i in range(n):
        if abs(a[i][i]) < 1e-12:
            raise ValueError(
                f"Диагональный элемент a[{i}][{i}] равен нулю. Метод Зейделя без перестановки не применим."
            )

    for iteration in range(1, max_iterations + 1):
        x_old = list(x)  # Сохраняем значения с предыдущей итерации

        for i in range(n):
            # Сумма с уже обновленными x_j (j < i)
            s1 = sum(a[i][j] * x[j] for j in range(i))

            # Сумма со старыми x_j (j > i)
            s2 = sum(a[i][j] * x_old[j] for j in range(i + 1, n))

            # Формула метода Зейделя
            x[i] = (b[i] - s1 - s2) / a[i][i]

        # Проверка условия сходимости (норма разности вектора |x^(k) - x^(k-1)|)
        max_diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if max_diff < eps:
            print(f"Сходимость достигнута за {iteration} итерации(ий).")
            return x

    raise RuntimeError(
        f"Метод не сошелся за максимальное число итераций ({max_iterations})."
    )


def main():
    input_filename = "input.txt"

    try:
        a, b = read_input(input_filename)
        n = len(b)
        print(f"Загружена система размерности: {n}x{n}\n")

        x = solve_seidel(a, b, eps=1e-6)

        print("\nРешение системы (x_1, x_2, ..., x_n):")
        for i, val in enumerate(x, 1):
            print(f"x_{i} = {val:.8f}")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_filename}' не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
