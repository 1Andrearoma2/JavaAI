package org.ScannerExample;

import ai.onnxruntime.*;
import org.example.Tokenizer;

import java.util.*;
import static org.example.Methods.*;

public class GPT2Java {
    public static Map<Integer, String> vocab;

    public static void main(String[] args) throws Exception {
        String vocabPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\vocab.json";
        String modelPath = "C:\\Users\\Andrearoma\\Desktop\\ai\\models\\TinyLlama\\onnx\\decoder_model.onnx";
        String[] states = {"Generazione risposta .  ", "Generazione risposta .. ", "Generazione risposta ..."};

        Scanner input = new Scanner(System.in);
        System.out.print("Inserire il prompt (solo inglese): ");
        String userPrompt = input.nextLine();
        System.out.print("Token utilizzabili: ");
        int maxTokens = input.nextInt();
        vocab = loadVocabulary(vocabPath);

        // Traduciamo il prompt in token leggibile dal modello
        long[] promptTokens = Tokenizer.tokenize(userPrompt);
        List<Long> inputTokens = new ArrayList<>();
        inputTokens.addAll(Arrays.stream(promptTokens).boxed().toList());

        // Crea sessione ONNX
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

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
            int i = (step + 1) % states.length;
            System.out.print("\r" + states[i]);
            if (nextToken == 50256) {
                break;
            }
        }

        // Estrai la risposta del modello ed esclude il prompt
        List<Long> generatedTokens = inputTokens.subList(promptTokens.length, inputTokens.size());
        String generatedText = decodeTokens(generatedTokens);
        generatedText = generatedText.replace("0x0A", " ").replace("Ċ", " ").replace("Ġ", " ").replace("< >", " ");

        System.out.println("\nDomanda: " + userPrompt + "\nRisposta: " + generatedText);
    }
}