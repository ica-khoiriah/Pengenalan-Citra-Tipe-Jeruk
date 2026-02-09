import textwrap
import sys

def app():
    from app import ready
    app = ready()
    app.launch()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(textwrap.dedent("""\
            Tidak ada yang bisa dilakukan -- Ketik Perintah:
            \t[t]rain\t\t- untuk melatih data.
            \t[a]pp\t\t- untuk menampilkan UI web.
             """))
        exit(0)

    from train import latih_model, SimpanModul
    from model_utils import muat_dataset
    from model import ModelCNNJeruk
    from plotting import (
        BahanPlottinganModel,
        siapkan_plottingan,
        plotting_loss_dan_latih,
    )
    _, command, *args = sys.argv

    match command:
        case "latih" | "train" | "t":
            model = ModelCNNJeruk(2)                                
            dataset_latih, dataset_validasi = muat_dataset()        
            simpan = SimpanModul(                                   
                nama="model-09_02_26_v2_14",
            )
            
            (daftar_akur_latih,
             daftar_akur_valid,
             daftar_loss_latih,
             daftar_loss_valid) = latih_model(                     
                model=model,
                pemuat_latih=dataset_latih,
                pemuat_validasi=dataset_validasi,
                simpan=simpan
            )

            bahan_plottingan_model = BahanPlottinganModel(          
                akurasi_latihan=daftar_akur_latih,
                akurasi_validasi=daftar_akur_valid,
                loss_latihan=daftar_loss_latih,
                loss_validasi=daftar_loss_valid
            )
            siapkan_plottingan(bahan_plottingan_model)              

            plotting_loss_dan_latih()
        case "aplikasi" | "app" | "a":
            app()
        case _:
            print("Perintah Salah. [latih/train, aplikasi/app]")