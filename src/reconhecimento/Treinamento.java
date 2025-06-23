package reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;



public class Treinamento {
    public static void main(String[] args) {
        File diretorio = new File ("src\\fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String nome) {
                return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png");
            }
        };
        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new MatVector(arquivos.length);
        Mat rotulos = new Mat (arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        for(File imagem: arquivos){
            Mat foto = opencv_imgcodecs.imread(imagem.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            resize(foto, foto, new Size (160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }
        FaceRecognizer lbph = LBPHFaceRecognizer.create();
        
        lbph.train(fotos, rotulos);
        lbph.save("src\\recursos\\classificadorLBPH.yml");

}
    
}

