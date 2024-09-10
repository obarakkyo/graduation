import numpy as np
import matplotlib.pyplot as plt

def main():
    index_list = np.array([i for i in range(10, 150, 5)])
    simple_buket = np.array([93, 82, 47, 54, 45, 31, 45, 47, 40, 18, 18, 30, 28, 18, 20, 24, 21, 19, 22, 6, 2, 6, 10, 32, 32, 32, 32, 32])
    nextto_buket = np.array([73, 78, 56, 71, 44, 38, 22, 30, 36, 29, 18, 21, 28, 27, 21, 27, 28, 8, 23, 18, 17, 19, 27, 14, 29, 19, 18, 16])
    position_buket = np.array([4, 0, 2, 0, 4, 0, 0, 4, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0])
    squared_buket = np.array([91, 90, 61, 45, 48, 36, 40, 30, 31, 44, 20, 28, 25, 18, 22, 11, 15, 33, 12, 17, 16, 22, 10, 14, 12, 12, 12, 18])
    xore_buket = np.array([111, 77, 50, 55, 41, 42, 44, 22, 28, 42, 25, 32, 34, 35, 34, 32, 34, 34, 36, 37, 32, 32, 32, 32, 32, 32, 32, 32])

    print(index_list)

    # plt.plot(index_list)
    plt.title("Confflict Count")
    plt.plot(index_list, simple_buket, label="Simple")
    plt.plot(index_list, nextto_buket, label="Nextto")
    plt.plot(index_list, position_buket, label="position")
    plt.plot(index_list, squared_buket, label="squared")
    plt.plot(index_list, xore_buket, label="xor")

    
    plt.legend()
    plt.savefig("../pictures/dataset4_confflict_plot.png")
    plt.show()



    

if __name__ == "__main__":
    main()