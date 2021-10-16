from main import main
from SMT_Solver.Config import config


if __name__ == "__main__":

    result_file = open(r"Result/RL_Prunning_NL.txt", "w")
    benchmarks_c_path = r"Benchmarks/NL/c/"
    benchmarks_g_path = r"Benchmarks/NL/c_graph/"
    benchmarks_s_path = r"Benchmarks/NL/c_smt/"
    results = []
    for i in range(1,31):
        gfilename = "NL" + str(i) + ".c.json"
        path_g = benchmarks_g_path + gfilename
        sfilename = "NL" + str(i) + ".c_smt"
        path_s = benchmarks_s_path + sfilename
        cfilename = "NL" + str(i) + ".c"
        path_c = benchmarks_c_path + cfilename
        timeUsed, answer = main(path_c, path_g, path_s)
        results.append(timeUsed)
        result_file.write(str(timeUsed) + '\t' + str(answer) + '\n')
        result_file.flush()

    result_file.close()
    print(results)
