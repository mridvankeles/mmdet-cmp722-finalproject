import subprocess

def run_sh(script):
    print(f"{script} çalıştırılıyor...")
    result = subprocess.run(["bash", script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"{script} çalışırken hata oluştu:\n", result.stderr)
        exit(1)
    else:
        print(f"{script} başarıyla tamamlandı.\n")

run_sh("psnr_train.sh")
run_sh("finetune_train.sh")
