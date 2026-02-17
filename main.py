from textwrap import dedent
from pathlib import Path
import sys

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(dedent("""\
            Tidak ada yang bisa dilakukan -- Ketik Perintah:
            \t[t]rain\t\t- untuk melatih data.
            \t[a]pp\t\t- untuk menampilkan UI web.
            \tt[e]st\t\t- untuk memprediksi data pada folder.
            \t[h]elp\t\t- untuk menampilkan pesan bantu.
             """))
        exit(0)

    from confmatrix import kalkulasi_konfusi_matrix
    from model_utils import (
        muat_dataset,
        muat_model,
        muat_data
    )
    from model import ModelCNNJeruk
    from train import latih_model
    from app import ready
    from local_const import FOLDER_MODEL
    from utils import ambil_file_terbaru
    from predict_loop import prediksi_banyak
    from plotting import (
        tampilkan_plot_konfusi_matrix,
        tampilkan_plot_akur_loss_data,
        tampilkan_plot_akur_loss,
        tampilkan_plot
    )
    _, command, *args = sys.argv

    match command:
        case "latih" | "train" | "t":
            jumlah_epoch: int | str | None = None
            if len(args) >= 1:
                command, *args = args

                match command:
                    case "epoch" | "^":
                        if len(args) < 1:
                            print("Jumlah epoch tidak memadai.")

                        jumlah_epoch, *args = args
                        if jumlah_epoch == "default" or jumlah_epoch == "biasa":
                            jumlah_epoch = 10
                        else:
                            jumlah_epoch = int(jumlah_epoch)
                    case "?":
                        print("\tpython main.py train epoch <jumlah_epoch:int>")
                        exit(0)
                    case _:
                        print("Perintah Salah. [epoch]")
                        exit(1)

            model = ModelCNNJeruk(2)                                
            dataset_latih, dataset_validasi = muat_dataset()        
            
            hasil_latih = latih_model(                     
                model=model,
                pemuat_latih=dataset_latih,
                pemuat_validasi=dataset_validasi,
                simpan=True,
                jumlah_epoch=jumlah_epoch
            )

            hasil_konfusi_matriks = kalkulasi_konfusi_matrix(
                hasil_latih
                .bahan_plottingan_model
                .plottingan_prediksi
            )

            tampilkan_plot_akur_loss(
                hasil_latih.bahan_plottingan_model,
                hasil_latih.nama_model
            )
            tampilkan_plot_konfusi_matrix(
                hasil_konfusi_matriks,
                hasil_latih.nama_model
            )
            tampilkan_plot()
        case "aplikasi" | "app" | "a":
            path: Path | None = None
            if len(args) >= 1:
                command, *args = args

                match command:
                    case "muat" | "load" | "->":
                        if len(args) < 1:
                            print("Tidak ada yang bisa di muat.")
                            exit(1)

                        path, *args = args
                        if path == "newest" or path == "terbaru" or path == "default":
                            path = ambil_file_terbaru(FOLDER_MODEL)
                        else:
                            path = Path(path)
                    case "?":
                        print("\tpython main.py app load <path_file:str>")
                        exit(0)
                    case _:
                        print("Perintah Salah. [load]")
                        exit(1)
            if path is None:
                path = ambil_file_terbaru(FOLDER_MODEL)
            app = ready(path)
            app()
        case "test" | "uji" | "u" | "e" | "+":
            path: Path | None = None

            if len(args) >= 1:
                path, *args = args
                path = Path(path)
                path_model = None

                if path.is_file():
                    print("Path bukan folder.")
                    exit(1)
                
                command, *args = args

                match command:
                    case "muat" | "load" | "->":
                        if len(args) < 1:
                            print("Tidak ada yang bisa di muat.")
                            exit(1)

                        path_model, *args = args
                        if path_model == "newest" or path_model == "terbaru" or path == "default":
                            path_model = ambil_file_terbaru(FOLDER_MODEL)
                        else:
                            path_model = Path(path)
                    case "?":
                        print("\tpython main.py test <path_folder:str> load <path_file:str>")
                        exit(0)
                    case _:
                        print("Perintah Salah. [load]")
                        exit(1)

                if path is None:
                    print("Path ke folder tidak ada.")
                    exit(1)

                if path_model is None:
                    print("Model tidak ada.")
                    exit(1)

                model = ModelCNNJeruk(2)
                muat_model(model, path_model)
                data = muat_data(path)

                hasil_predik = prediksi_banyak(
                    model=model,
                    pemuat_data=data,
                    nama_prediksi=f"{path_model.stem}_dengan_Data_{path.stem.title()}"
                )

                hasil_konfusi_matriks = kalkulasi_konfusi_matrix(
                    hasil_predik
                    .bahan_plottingan_data
                    .plottingan_prediksi
                )

                tampilkan_plot_akur_loss_data(
                    hasil_predik.bahan_plottingan_data,
                    hasil_predik.nama_prediksi
                )
                tampilkan_plot_konfusi_matrix(
                    hasil_konfusi_matriks,
                    hasil_predik.nama_prediksi
                )
                tampilkan_plot()
        case "help" | "h" | "?":
            print(dedent("""
            Perintah yang bisa dijalankan:
            \t[t]rain\t\t- untuk melatih model. subperintah opsional:
            \t epoch <int>\t- untuk menspesifikasikan jumlah epoch. [default] 10.
            \n
            \t[a]pp\t\t- untuk menampilkan GUI model pada Web. subperintah opsional:
            \t load <path>\t- untuk menspesifikasikan model yang akan dipakai. [default] model terbaru.
            \n
            \tt[e]st <path>\t- untuk menguji pada sekumpulan citra. subperintah wajib:
            \t load <path>\t- untuk menspesifikasikan model yang akan dipakai. [default] model terbaru.
            \n
            \t[h]elp\t\t- untuk menampilkan pesan bantu seperti ini.
            """))
            exit(0)
        case _:
            print("Perintah Salah. [train, app]")
            exit(1)