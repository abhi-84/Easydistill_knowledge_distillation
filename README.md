This project was implemented to validate the Easydistill framework for Knowledge Distillation.
Here, in Easydistill_project/easydistill/recipes/distilqwen_series, DistilQwen2.5 is used, in which model Qwen2.5-7B is used as the Teacher model and 
Qwen2.5-0.5B is used as the student model. 

During Black Box testing (Stage 1), the student model is trained with the distillqwen-100k dataset. This dataset is available 
on HuggingFace Hub. This dataset consists of the Teacher model output for a given input.
Due to this, for Stage 1, the Teacher model is not required, and the dataset alone can do. Once training is completed, student model is saved.

During White Box testing(Stage 2), the student model (now, the student model is the result saved from stage 1), 
is trained using a logit file generated from the Teacher model. This logit file helps the student model to understand the reasonings
done by the Teacher model to come to an output conclusion for given inputs, also called Chain of Thoughts (COT)
Here, the teacher model needs to be loaded locally alongside the student as API don't expose logits

Model Training resource requirements: To train the student model, it requires High CPU/GPU, RAM and disk space. I have used Google Colab Pro V100 GPU; 40GB RAM and 120GB+ disk space

The process:
1. Training batch is fed to both models simultaneously 
2. Teacher produces logits (probability distributions) 
3. Student produces logits 
4. KL divergence loss computed between them 
5. Only student weights updated (teacher remains frozen)

Steps to implement

1. git clone https://github.com/modelscope/easydistill
2. cd easyDistill
3. pip3 install -e .
4. Download DistilQwen_100K dataset https://huggingface.co/datasets/alibaba-pai/DistilQwen_100k/tree/main/data
5. Convert dataset from parquet format to json format par2json.py
6. Correctly configure distilqwen2.5_stage1.json to set student model path, dataset path and result path
7. pip3 install jsonlines
8. Train the student model using Black-box KD

   python3 easydistill/kd/train.py --config ./recipes/distilqwen_series/distillqwen2.5/distilqwen2.5_stage1.json
   
10. Save model state as checkpoints to be used
11. Test the saved model checkpoint using file test_model_interactive.py
12. Download the 7B teacher model from the Hugging Face Hub
13. Split the dataset-100k into multiple chunks of 20K each to save memory space, split_dataset.py
14. Create 5 config files for each database chunk split_json.py
15. Use a config file for each chunk and fo inference to generate logits for each chunk

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/infer.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/stage2_chunk_0.json

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/infer.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/stage2_chunk_1.json

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/infer.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/stage2_chunk_2.json

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/infer.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/stage2_chunk_3.jso

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/infer.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/stage2_chunk_4.json
    
17. Merging all logits into a single logit file merge_logit.py
18. Perform the student model training using logit file and the stage2.json config file. Update the stage2.json file with correct configuration values, logits.json path, student model and teacher model path and path for result

    python3 /content/drive/MyDrive/easydistill/easydistill/kd/train.py --config /content/drive/MyDrive/easydistill/recipes/distilqwen_series/distillqwen2.5/distilqwen2.5_stage2.json
    
20. Test the trained student model using test_model_interactive.py
21. Run Flask server, host the front-end UI and host the model
