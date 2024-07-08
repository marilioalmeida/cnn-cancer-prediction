import subprocess
import os

# Desativar todos os logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Definir diretórios
cnns_dir = os.path.abspath("./cnns")
code_share_dir = os.path.abspath("./code_share")

# Definir script de embaralhamento
shuffle_script = os.path.join(code_share_dir, "ShuffleFileOrder_and_GenerateNpyFile.py")

# Lista de scripts para execução
scripts = [
    os.path.join(cnns_dir, "alexnet.py"),
    os.path.join(cnns_dir, "cnn.py"),
    os.path.join(cnns_dir, "cnnlayer2.py"),
    os.path.join(cnns_dir, "densenet.py"),
    os.path.join(cnns_dir, "inceptionv3.py"),
    os.path.join(cnns_dir, "lenet.py"),
    os.path.join(cnns_dir, "resnet50.py"),
    os.path.join(cnns_dir, "vgg16.py"),
    os.path.join(cnns_dir, "vgg19.py"),
    os.path.join(cnns_dir, "xception.py")
]

num_executions = 50

# Loop de execução
for i in range(num_executions):
    print(f"Starting execution cycle {i+1}/{num_executions}")

    try:
        # Executar o script ShuffleFileOrder_and_GenerateNpyFile.py
        print(f"Executing {shuffle_script}")
        subprocess.run(["python", shuffle_script], check=True)
        print(f"Executed {shuffle_script} {i+1}/{num_executions} times successfully")

        # Executar os outros scripts
        for script in scripts:
            print(f"Executing {script}")
            subprocess.run(["python", script], check=True)
            print(f"Executed {script} {i+1}/{num_executions} times successfully")
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {e.cmd}. Exit code: {e.returncode}")
    
    print(f"Completed execution cycle {i+1}/{num_executions}")

print("All scripts executed 50 times.")
