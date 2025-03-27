package org.example;

import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.Map;

public class TokenDecoder {

//    public static long[] Decoder(String prompt) throws IOException {
//        // Carica il vocabolario (file JSON)
//        String vocabPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\gpt2\\vocab.json";
//        String vocabJson = new String(Files.readAllBytes(Paths.get(vocabPath)));
//
//        // Usa Gson per convertire JSON in una mappa con valori numerici
//        Gson gson = new Gson();
//        Map<String, Double> vocab = gson.fromJson(vocabJson, Map.class);
//
//        // Token IDs
//
//        String[] tokenIds = prompt.split(" ");
//        long[] longArray = new long[tokenIds.length];
//        longArray[0] = 100;
//
//        // Decodifica gli ID dei token in parole (gestendo il valore numerico)
//        for (String tokenId : tokenIds) {
//            // Converte il valore numerico in stringa, se è un Double
//            Double value = vocab.get(tokenId);
//            System.out.println("Token ID " + tokenId + " decodificato in valore: " + value);
//            longArray[longArray.length] = Long.valueOf(tokenId);
//        }
//    }

    public static void main(String[] args) throws IOException{
        // Carica il vocabolario (file JSON)
        String prompt = "What is machine learning?";
        String vocabPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\gpt2\\vocab.json";
        String vocabJson = new String(Files.readAllBytes(Paths.get(vocabPath)));

        // Usa Gson per convertire JSON in una mappa con valori numerici
        Gson gson = new Gson();
        Map<String, Double> vocab = gson.fromJson(vocabJson, Map.class);

        // Token IDs

        String[] tokenIds = prompt.split(" ");
        long[] longArray = new long[tokenIds.length];
        longArray[0] = 100;

        // Decodifica gli ID dei token in parole (gestendo il valore numerico)
        for (String tokenId : tokenIds) {
            // Converte il valore numerico in stringa, se è un Double
//            Double value = vocab.get(tokenId);
//            System.out.println("Token ID " + tokenId + " decodificato in valore: " + value);
            longArray[longArray.length] = Long.valueOf(tokenId);
        }
    }
}
