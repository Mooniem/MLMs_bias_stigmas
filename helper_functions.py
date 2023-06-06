import os
import cairosvg
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, XLNetLMHeadModel, XLNetTokenizer
import numpy as np

def write_base_prompts(prompt_templates):
    '''
    Generate prompts for the base condition
    '''
    prompt_list = []
    for prompt in prompt_templates:
        prompt_new = prompt.replace(" who VERB/BE CONDITION","")
        prompt_list.append(prompt_new)
        print(prompt_new)
    return(prompt_list)

def write_prompts (dataset,prompt_templates,plural):
    """
    Generate prompts for each conditions
    """
    prompt_list = []
    paraphrase_list  = []
    pachankis_condition_list = []
    condition_list = []
    category_list = []
    for index, row in dataset.iterrows(): # for every condition 
        verb_be = row[0] #the value for the VERB/BE
        condition = row[2] #the value for the condition
        condition_plural=row[1]# the value for plural 
        category = row[3] #the value for the category 
        pachankis_condition = row[4] #the column for the pachankis category
        verb_be_plural=row[5] # the column for verb-ber

        verb_be_replacement = ""
        
        if plural == True:
            print(verb_be_plural)
            if verb_be_plural=="Verb":
                verb_be_replacement="have "
            elif verb_be_plural=="Be":
                verb_be_replacement="are "
            elif verb_be_plural=="None":
                verb_be_replacement=""
            else:
                verb_be_replacement= verb_be_plural + " "
        else:#if it is singular (someone)
            if verb_be == "Be":
                verb_be_replacement = "is "
            elif verb_be == "Verb":
                verb_be_replacement = "has "
            elif verb_be_plural=="None":
                verb_be_replacement=""
            else:
                verb_be_replacement= verb_be+" "

        for prompt in prompt_templates:
            prompt_new = prompt.replace("VERB/BE ",verb_be_replacement)
            if plural ==True and condition_plural!="none":
                prompt_new = prompt_new.replace("CONDITION",condition_plural)
            else:
                prompt_new = prompt_new.replace("CONDITION",condition)
            #Keep track of paraphrases
            paraphrase = prompt_new.split()[-3:]
            paraphrase = " ".join(paraphrase)
            prompt_list.append(prompt_new)
            condition_list.append(condition)
            pachankis_condition_list.append(pachankis_condition)
            category_list.append(category)
            paraphrase_list.append(paraphrase)
            print("Experiment Prompt: ",prompt_new)
    return (prompt_list,paraphrase_list,condition_list,pachankis_condition_list,category_list)

def get_MLM_predictions(Prompts,dataset,base,model,top_tokens):
    nlp = pipeline('fill-mask',model=model)
    prompts_list = []
    probs = []
    stigma_conditions = []
    predictions= []
    categories = []
    pachankis_conditions = []
    
    if base ==True:
        prompt_templates =[]
        for prompt in Prompts:
            prompt_new = prompt.replace(" who VERB/BE CONDITION","")
            print("Experiment Prompt--: ",prompt_new)
            #pass prompt to language model
            output = nlp(prompt_new,top_k = top_tokens) # top_k indicates top 10 predictions
            for per_output in output:
                prompts_list.append(prompt_new)
                prompt_templates.append(prompt)
                stigma_conditions.append("None")
                pachankis_conditions.append("None")
                categories.append("None")
                #print(per_output['token_str'])
                predictions.append(per_output['token_str'])
                probs.append(per_output['score'])
    else:

        for index, row in dataset.iterrows():
            verb_be = row[0] #the column for the VERB/BE
            print(verb_be)
            condition = row[2] #the column for the condition
            category = row[3] #the column for the category 
            pachankis_condition = row[4] #the column for the pachankis category
            
            verb_be_replacement = ""
            if verb_be == "Be":
                verb_be_replacement = "is "
            elif verb_be == "Verb":
                verb_be_replacement = "has "
            elif verb_be == "None":
                verb_be_replacement = ""
            else:
                print(verb_be)
                verb_be_replacement = verb_be +" "

            for prompt in Prompts:
                prompt_new = prompt.replace("VERB/BE ",verb_be_replacement)
                prompt_new = prompt_new.replace("CONDITION",condition)
                print("Experiment Prompt--: ",prompt_new)
                
                #pass prompt to language model
                output = nlp(prompt_new,top_k = top_tokens) # top_k indicates top 10 predictions
                for per_output in output:
                    prompts_list.append(prompt_new)
                    stigma_conditions.append(condition)
                    pachankis_conditions.append(pachankis_condition)
                    categories.append(category)
                    #print(per_output['token_str'])
                    predictions.append(per_output['token_str'])
                    # print(per_output['token_str'])
                    probs.append(per_output['score'])
                #print(f"Word:",per_output['token_str'],f"---Confidence Score:",round(per_output['score'],4))

    df = pd.DataFrame()
    df["prompt"]=prompts_list
    df["condition"]=stigma_conditions
    df["predicted_word"]=predictions
    df["probs"]=probs
    df["category"]=categories
    df["pachankis_conditions"]=pachankis_conditions
    if base==True:
        df["prompt_templates"]=prompt_templates
    return (df,top_tokens,model)

def xlnet_predict_topK(prompt_info):

    tokenizer =AutoTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', return_dict = True)

    #Read in prompt information 
    prompts =prompt_info[0]
    paraphrases = prompt_info[1]
    condition_list = prompt_info[2]
    pachankis_condition_list = prompt_info[3]
    category_list = prompt_info[4]

    #create lists to store informaiton of predictions 
    prompt_list = []
    predicted_word_list =[]
    probability_list = []

    condition = []
    pachankis_conditions = []
    category = []

    for index, prompt in enumerate(prompts):
      input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)  
      #"I gave you three apples. I have <mask> apples in hands"
      mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
      # print("mask_token_index: ",mask_token_index)
      targets = [mask_token_index]

      perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
      perm_mask[0, :, targets] = 1.0  # Previous tokens don't see last token

      target_mapping = torch.zeros((1, len(targets), input_ids.shape[1]), dtype=torch.float)  

      target_mapping[0, 0, targets[0]] = 1.0  # Our first  prediction 
      #target_mapping[0, 1, targets[1]] = 1.0  # Our second  prediction 

      model.eval()
      if torch.cuda.is_available(): model.to('cuda') #if we have a GPU 

      with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

      # print(next_token_logits.shape)
      # print(next_token_logits[0][0].shape)
      convert_logits_to_probs = torch.softmax(next_token_logits[0],dim =1) #convert logits to probs
      # print("probability list: ",convert_logits_to_probs)
      # print("printing sum probs...", torch.sum(convert_logits_to_probs))
      probs_list = convert_logits_to_probs[0,:].numpy()

      for j in range(len(targets)):
        predicted_k_indexes = torch.topk(outputs[0][0][j],k=50) # obtain the indexes of top 10 words 
        predicted_logits_list = predicted_k_indexes[0] # 
      # print(predicted_logits_list)
        predicted_indexes_list = predicted_k_indexes[1] 
      
      # print ("predicted word:",tokenizer.decode(input_ids[0][targets[j]].item()), j)
        for i,item  in enumerate(predicted_indexes_list):
            the_index = predicted_indexes_list[i].item()
            word_logit = predicted_logits_list[i].item()
            word = tokenizer.decode(the_index)
            word_prob = probs_list[the_index]
            print("word and logits and prob---",word,word_logit,word_prob)  
            prompt_list.append(prompt)  
            predicted_word_list.append(word)
            probability_list.append(word_prob)
            print(condition_list[index])
            condition.append(condition_list[index])
            print(pachankis_condition_list[index])
            pachankis_conditions.append(pachankis_condition_list[index])
            category.append(category_list[index])
            
      
    df= pd.DataFrame()
    df["prompt"]=prompt_list 
    df["predicted_word"] = predicted_word_list
    df["probs"] = probability_list
    df["condition"] = condition
    df["pachankis_conditions"]=pachankis_conditions
    df["category"]=category
    return df


def xlnet_predict_topK_base(prompts):
    '''
    This function takes in a list of prompts and returns a dataframe with the top 50 predictions for each prompt.
    
    Note that this function is different from the xlnet_predict_topK function in that it does not take in any information about conditions.

    Need to modify k value if needs more predictions other than the top 50 predictions.
    '''

    tokenizer =AutoTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', return_dict = True)

    #create lists to store informaiton of predictions 
    prompt_list = []
    predicted_word_list =[]
    probability_list = []


    for index, prompt in enumerate(prompts):
      
      input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)  
      mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
      targets = [mask_token_index]

      perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
      perm_mask[0, :, targets] = 1.0  # Previous tokens don't see last token

      target_mapping = torch.zeros((1, len(targets), input_ids.shape[1]), dtype=torch.float)  
      target_mapping[0, 0, targets[0]] = 1.0  # Our first  prediction 

      model.eval()
      if torch.cuda.is_available(): model.to('cuda') #if we have a GPU 

      with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

      convert_logits_to_probs = torch.softmax(next_token_logits[0],dim =1) #convert logits to probs

      probs_list = convert_logits_to_probs[0,:].numpy()

      for j in range(len(targets)):
        predicted_k_indexes = torch.topk(outputs[0][0][j],k=50) # obtain the indexes of top 50 words 
        predicted_logits_list = predicted_k_indexes[0] # 
        predicted_indexes_list = predicted_k_indexes[1] 
      
        for i,item  in enumerate(predicted_indexes_list):
            the_index = predicted_indexes_list[i].item()
            word_logit = predicted_logits_list[i].item()
            word = tokenizer.decode(the_index)
            word_prob = probs_list[the_index]
            print("word and logits and prob---",word,word_logit,word_prob)  
            prompt_list.append(prompt)  
            predicted_word_list.append(word)
            probability_list.append(word_prob)
    
    df= pd.DataFrame()
    df["prompt"]=prompt_list 
    df["predicted_word"] = predicted_word_list
    df["probs"] = probability_list
    return df


def run_sentiment_analysis(model_name,prompt_info):

    sentiment_analysis = pipeline("sentiment-analysis",model=model_name)

    prompts =prompt_info[0]
    condition_list = prompt_info[2]
    pachankis_condition_list = prompt_info[3]
    category_list = prompt_info[4]

    #create lists to store informaiton of predictions 
    prompt_list = []
    sentiment_list =[]
    score_list = []

    condition = []
    pachankis_conditions = []
    category = []
    prompts= prompt_info[0]

    for index, prompt in enumerate(prompts):
        print(prompt)
        prompt_list.append(prompt)
        result = sentiment_analysis(prompt)
        sentiment = result[0]['label']
        sentiment_score = result[0]['score']
        sentiment_list.append(sentiment)
        score_list.append(sentiment_score)
        condition.append(condition_list[index])
        pachankis_conditions.append(pachankis_condition_list[index])
        category.append(category_list[index])

    
    df = pd.DataFrame()
    df["prompts"]=prompt_list
    df["sentiment"] = sentiment_list
    df["sentiment_score"]=score_list
    df["condition"] = condition
    df["pachankis_conditions"]=pachankis_conditions
    df["category"] = category

    return df

def save_results (results,results_name,social_distance,condition):

    if social_distance ==True:
        if condition =="stigma":
            filename = results_name + "_stigma_SD.csv"
        elif condition =="nonstigma":
            filename = results_name + "_nonstigma_SD.csv"
        else:
            filename = results_name+"_SD.csv"
    else:
        filename = results_name +".csv"

    results.to_csv(filename)

def convert_svg_to_png(folder_path):
    '''
    Converts all SVG files in a folder to PNG files
    :param folder_path: Path to the folder containing SVG files
    :return: None
    '''
    # Specify the path to the folder containing SVG files

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter the files to include only SVG files
    svg_files = [file for file in files if file.endswith('.svg')]

    # Print the list of SVG files
    for svg_file in svg_files:
        filename = os.path.splitext(svg_file)[0]
        print(filename)

        png_file = os.path.join(folder_path, filename + '.png')
        cairosvg.svg2png(url=svg_file, write_to=png_file)


