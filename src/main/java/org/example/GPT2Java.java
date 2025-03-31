package org.example;

import ai.onnxruntime.*;
import java.util.*;
import static org.example.Methods.*;

public class GPT2Java {
    public static Map<Integer, String> vocab;

    public static void main(String[] args) throws Exception {
        String vocabPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\vocab.json";
        String modelPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\onnx\\decoder_model.onnx";

        String userPrompt = "What is machine learning?";
        vocab = loadVocabulary(vocabPath);

        // Traduciamo il prompt in token leggibile dal modello
        long[] promptTokens = Tokenizer.tokenize(userPrompt);
        List<Long> inputTokens = new ArrayList<>();
        inputTokens.addAll(Arrays.stream(promptTokens).boxed().toList());

        int maxTokens = 30;  // Numero massimo di token generati
        long eosToken = 50256;

        // Crea sessione ONNX
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

        // Per ogni token da generare esegue questo ciclo
        for (int step = 0; step < maxTokens; step++) {
            long[] inputArray = inputTokens.stream().mapToLong(Long::longValue).toArray(); // Contiene i token input
            long[] attentionMaskArray = new long[inputArray.length]; // Comunica quali token da considerare e quali fungono da padding
            Arrays.fill(attentionMaskArray, 1);

            // Trasformiamo gli array in tensor leggibili dal modello
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, new long[][]{inputArray});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, new long[][]{attentionMaskArray});

            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", inputTensor);
            feeds.put("attention_mask", attentionMaskTensor);
            OrtSession.Result result = session.run(feeds);

            float[][][] outputData = (float[][][]) result.get(0).getValue(); // Contiene tutte le previsione che ha fatto
            float[] logits = outputData[0][outputData[0].length - 1]; // Contiene tutti i punteggi del token
            int nextToken = argmax(applySoftmax(logits)); // Ricorda il token piú significativo

            inputTokens.add((long) nextToken); // Aggiunge il token piú significativo nell'array finale
            String tokenToString = decodeToken(nextToken); // Decodifica il token in stringa
            tokenToString = tokenToString.replace("0x0A", " ").replace("Ġ", " ").replace("Ċ", " ");
            System.out.println("Token generato: " + nextToken + " → " + tokenToString);

            if (nextToken == eosToken) {
                break;
            }
        }

        // Estrai la risposta del modello ed esclude il prompt
        List<Long> generatedTokens = inputTokens.subList(promptTokens.length, inputTokens.size());
        String generatedText = decodeTokens(generatedTokens);
        generatedText = generatedText.replace("0x0A", " ").replace("Ċ", " ").replace("Ġ", " ");

        System.out.println("\n" + userPrompt + "\n" + generatedText);
    }
}