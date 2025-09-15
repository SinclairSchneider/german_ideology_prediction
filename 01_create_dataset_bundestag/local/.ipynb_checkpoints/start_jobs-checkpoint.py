from subprocess import Popen

def main():
    i_total_processes = 4
    for i in range(i_total_processes): #GPU
        Popen(["python", "local_classification.py", str(i), str(i_total_processes)])

if __name__ == "__main__":
    main()