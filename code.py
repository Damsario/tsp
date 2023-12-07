import numpy as np
import time
import random
import sys
import itertools
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.pyplot as plt

def get_distance_matrix(num_cities):
    distances = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            while True:
                try:
                    distance = float(input(f"Введите расстояние от города {i} до города {j}: "))
                    distances[i][j] = distances[j][i] = distance
                    break
                except ValueError:
                    print("Пожалуйста, введите действительное число.")
    return distances

def plot_tsp_solution(path, distances):
    num_cities = len(distances)
    angles = np.linspace(0, 2 * np.pi, num_cities, endpoint=False).tolist()
    x = np.cos(angles)
    y = np.sin(angles)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    for i, txt in enumerate(range(num_cities)):
        plt.annotate(txt, (x[i], y[i]), fontsize=12, ha='right')

    for i in range(num_cities):
        start_pos = path[i]
        end_pos = path[(i + 1) % num_cities]
        plt.arrow(x[start_pos], y[start_pos], x[end_pos] - x[start_pos], y[end_pos] - y[start_pos],
                  length_includes_head=True, head_width=0.03, head_length=0.05, fc='lightblue', ec='black')
        plt.text((x[start_pos] + x[end_pos]) / 2, (y[start_pos] + y[end_pos]) / 2, str(i),
                 fontsize=9, color='red')

    plt.title('Оптимальный маршрут коммивояжера')
    plt.axis('equal')
    plt.show()

def branch_and_bound_tsp(distances):
    def tsp(curr_pos, visited, curr_length, count, path):
        nonlocal min_length, optimal_route

        if count == n:
            curr_length += distances[curr_pos][0]
            if curr_length < min_length:
                min_length = curr_length
                optimal_route = path[:]
            return

        for i in range(n):
            if not visited[i] and distances[curr_pos][i]:
                visited[i] = True
                path[count] = i
                tsp(i, visited, curr_length + distances[curr_pos][i], count + 1, path)
                visited[i] = False

    start_time = time.perf_counter()

    n = len(distances)
    min_length = sys.maxsize
    optimal_route = None
    visited = [False] * n
    visited[0] = True
    path = [None] * n
    path[0] = 0

    tsp(0, visited, 0, 1, path)

    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000 # Преобразование в миллисекунды

    return min_length, execution_time, optimal_route

def solve_tsp(distances):
    distance_matrix = np.array(distances)
    start_time = time.perf_counter()
    path, distance = solve_tsp_dynamic_programming(distance_matrix)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000
    return path, distance, execution_time

def main():
    while True:
        try:
            num_cities = int(input("Введите количество городов: "))
            if num_cities <= 0:
                raise ValueError
            break
        except ValueError:
            print("Пожалуйста, введите положительное целое число.")

    distances = get_distance_matrix(num_cities)

    # Выбор алгоритма в зависимости от количества городов
    if num_cities <= 6:
        min_length, execution_time, optimal_route = branch_and_bound_tsp(distances)

    else:
        path, distance, execution_time = solve_tsp(distances)
        optimal_route = path
        min_length = distance
        3

    # Вывод результатов и визуализация
    print("*" * 40)
    print("Длина оптимального маршрута: {:.2f}".format(min_length))
    print("Время выполнения: {:.5f} миллисекунд".format(execution_time))
    print("Оптимальный маршрут:", optimal_route)
    print("*" * 40 + "\n")
    plot_tsp_solution(optimal_route, distances)

if __name__ == "__main__":
    main()
