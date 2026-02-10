from pathlib import Path
import textwrap
import sys

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(textwrap.dedent("""\
            Tidak ada yang bisa dilakukan -- Ketik Perintah:
            \t[t]rain\t\t- untuk melatih data.
            \t[a]pp\t\t- untuk menampilkan UI web.
             """))
        exit(0)

    from model_utils import muat_dataset
    from utils import ambil_file_terbaru
    from model import ModelCNNJeruk
    from train import latih_model
    from app import ready
    from local_const import FOLDER_MODEL
    from plotting import (
        BahanPlottinganModel,
        BahanPlottingan,
        tampilkan_plot_akur_loss,
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
                    case _:
                        print("Perintah Salah. [epoch]")
                        exit(1)

            model = ModelCNNJeruk(2)                                
            dataset_latih, dataset_validasi = muat_dataset()        
            
            (daftar_akur_latih,
             daftar_akur_valid,
             daftar_loss_latih,
             daftar_loss_valid,
             nama_model) = latih_model(                     
                model=model,
                pemuat_latih=dataset_latih,
                pemuat_validasi=dataset_validasi,
                simpan=True,
                jumlah_epoch=jumlah_epoch
            )

            plot_loss = BahanPlottingan(
                latihan=daftar_loss_latih,
                validasi=daftar_loss_valid,
            )
            plot_akur = BahanPlottingan(
                latihan=daftar_akur_latih,
                validasi=daftar_akur_valid,
            )
            bahan_plottingan_model = BahanPlottinganModel(          
                plottingan_akurasi=plot_akur,
                plottingan_loss=plot_loss,
            )

            tampilkan_plot_akur_loss(bahan_plottingan_model, nama_model)
        case "aplikasi" | "app" | "a":
            path: Path | None = None
            if len(args) >= 1:
                command, *args = args

                match command:
                    case "muat" | "load" | "->":
                        if len(args) < 1:
                            print("Tidak ada yang bisa di muat.")

                            path, *args = args
                            if path == "newest" or path == "terbaru":
                                path = ambil_file_terbaru(FOLDER_MODEL)
                            else:
                                path = Path(path)
                    case _:
                        print("Perintah Salah. [load]")
            if path is None:
                path = ambil_file_terbaru(FOLDER_MODEL)
            app = ready(path)
            app.launch()
        case _:
            print("Perintah Salah. [train, app]")
            exit(1)