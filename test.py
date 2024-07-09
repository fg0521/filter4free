import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import RoFormerForSequenceClassification, RoFormerTokenizer
import torch

def classifier():
    df = pd.read_excel('/Users/maoyufeng/Downloads/all_data.xlsx')
    for tag in [0, 1]:
        df_sample = df[df['tag'] == tag]
        df1 = df_sample[df_sample['label_name'] == '其它']
        df2 = df_sample[df_sample['label_name'] == '涉黄']
        df3 = df_sample[df_sample['label_name'] == '辱骂']
        df4 = df_sample[df_sample['label_name'] == '涉恐涉暴']
        df5 = df_sample[df_sample['label_name'] == '涉政']
        df1.to_csv(f'/Users/maoyufeng/Downloads/违规数据集/其它_{tag}.csv', index=False, encoding='utf-8')
        df2.to_csv(f'/Users/maoyufeng/Downloads/违规数据集/涉黄_{tag}.csv', index=False, encoding='utf-8')
        df3.to_csv(f'/Users/maoyufeng/Downloads/违规数据集/辱骂_{tag}.csv', index=False, encoding='utf-8')
        df4.to_csv(f'/Users/maoyufeng/Downloads/违规数据集/涉恐涉暴_{tag}.csv', index=False, encoding='utf-8')
        df5.to_csv(f'/Users/maoyufeng/Downloads/违规数据集/涉政_{tag}.csv', index=False, encoding='utf-8')


def infer(df):
    # 加载预训练的 RoFormer 模型和分词器
    model_name = '/Users/maoyufeng/Downloads/违规数据集checkpoint-best'
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    model = RoFormerForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # df = pd.read_csv('/Users/maoyufeng/Downloads/model_CAC_illegal_3_A_20240703101737.csv')
    # df['text'] = df['clue_data'].apply(lambda x: list(eval(x).values())[0])
    # df = df[['text']]
    text = df['text'].tolist()
    labels = []
    # 使用分词器编码文本并进行padding
    for i in tqdm(range(0, len(text), 16)):
        inputs = tokenizer(text[i:i + 16], return_tensors="pt", padding=True)
        # 进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            outputs = torch.softmax(outputs.logits,dim=1).detach().cpu().numpy()
            # outputs = outputs.logits.detach().cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            labels.extend(list(outputs))
    df['pred'] = labels
    df['diff'] = df.apply(lambda x: 1 if x['label'] == x['pred'] else 0, axis=1)
    print(df)
    # df.to_csv('/Users/maoyufeng/Downloads/pred3.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    # df = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/辱骂_1.csv')
    # df.fillna(value='',inplace=True)
    # keywords = []
    # for keyword in df['keyword'].values.tolist():
    #     if keyword:
    #         keywords.extend(keyword.split('、'))
    # keywords = list(set(keywords))
    #
    # def get_words(s):
    #     res = [k for k in keywords if k in s.lower()]
    #     return '、'.join(res)
    # df['keyword'] = df['text'].apply(lambda x:get_words(x))
    # df_train = df[['text','keyword']]
    # df0 = df_train[df_train['keyword']==''].sample(1000)
    # df0['label']=0
    # df0.drop('keyword',axis=1,inplace=True)
    # def add_noise(x1,x2):
    #     if x2:
    #         return x1
    #     else:
    #         key = random.choice(keywords)
    #         pos = random.randint(0,len(x1)-1)
    #         return x1[:pos]+key+x1[pos:]
    #
    # df_train['text'] = df_train.apply(lambda x:add_noise(x['text'],x['keyword']),axis=1)
    # df_train['label']=1
    # df_train.drop('keyword',axis=1,inplace=True)
    #
    # num1 = int(len(df_train)*0.1)
    # num0 = int(len(df0)*0.1)
    # df_val = pd.concat([df_train[:num1],df0[:num0]],axis=0)
    # df_train = pd.concat([df_train[num1:],df0[num0:]],axis=0)
    # df_val = df_val.sample(frac=1)
    # df_train = df_train.sample(frac=1)
    # df_train.to_csv('/Users/maoyufeng/Downloads/违规数据集/train.csv',index=False,encoding='utf-8')
    # df_val.to_csv('/Users/maoyufeng/Downloads/违规数据集/val.csv',index=False,encoding='utf-8')

    # df1 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/谩骂1.csv')
    # df2 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/谩骂2.csv')
    # df = pd.concat([df1,df2],axis=0)
    # df['text'] = df['text'].apply(lambda x:x.lower())
    # df.drop_duplicates(keep='first',subset=['text'],inplace=True)
    # df.to_csv('/Users/maoyufeng/Downloads/违规数据集/谩骂.csv',index=False,encoding='utf-8')
    # print(df)

    # df = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/谩骂.csv')
    # df_add = pd.read_csv('/Users/maoyufeng/Downloads/pred2.csv')
    # df = pd.concat([df,df_add],axis=0)
    # df = df.sample(frac=1)
    # df1 = df[df['label'] == 1]
    # df0 = df[df['label'] == 0]
    # num1,num0 = int(len(df1)*0.1),int(len(df0)*0.1)
    # df_train = pd.concat([df1[num1:],df0[num0:]])
    # df_val = pd.concat([df1[:num1],df0[:num0]])
    # df_train = df_train.sample(frac=1)
    # df_val = df_val.sample(frac=1)
    # df_train.to_csv('/Users/maoyufeng/Downloads/违规数据集/train.csv',index=False,encoding='utf-8')
    # df_val.to_csv('/Users/maoyufeng/Downloads/违规数据集/val.csv',index=False,encoding='utf-8')

    df = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/val.csv')
    df.drop_duplicates(keep='first',subset='text',inplace=True)
    df.dropna(inplace=True,axis=0)
    infer(df)

    # df = pd.read_csv('/Users/maoyufeng/Downloads/`clue_judge`_20240703173556.csv')
    # df.drop_duplicates(subset='clue_data',inplace=True)
    # df['text'] = df['clue_data'].apply(lambda x: str(x))
    # df = df[['text']]
    # df = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/val.csv')
    # df.columns = ['text','true_label']



    # df = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/涉黄_1.csv')
    # df['len'] = df['text'].apply(lambda x:len(str(x)))
    # df = df[(df['len']>20) & (df['len'] < 100)]
    # df = df[df['conf']>0.95]
    # df = df.sample(1100)
    # df = df[['text','label']]
    # df['label']=1
    # df1 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/涉黄_0.csv')
    # df1['len'] = df1['text'].apply(lambda x: len(str(x)))
    # df1 = df1[(df1['len'] > 20) &( df1['len'] < 100)]
    # df3 = df1[df1['conf']<0.65]
    # df3 = df3.sample(2200)
    # df3 = df3[['text', 'label']]
    # df3['label']=0
    # df2 = df1[df1['conf']>0.8]
    # df2 = df2.sample(1100)
    # df2 = df2[['text', 'label']]
    # df2['label'] = 1
    #
    # df_train = pd.concat([df[100:],df2[100:],df3[200:]],axis=0)
    # df_val= pd.concat([df[:100],df2[:100],df3[:200]],axis=0)
    #
    # df_train = df_train.sample(frac=1)
    # df_val = df_val.sample(frac=1)
    # df_train.to_csv('/Users/maoyufeng/Downloads/违规数据集/train_yellow.csv',index=False,encoding='utf-8')
    # df_val.to_csv('/Users/maoyufeng/Downloads/违规数据集/val_yellow.csv',index=False,encoding='utf-8')
    # print(df)


    # df1 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/train_yellow.csv')
    # df2 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/train_revile.csv')
    #
    # df3 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/val_yellow.csv')
    # df4 = pd.read_csv('/Users/maoyufeng/Downloads/违规数据集/val_revile.csv')
    # df3['label'] = df3['label'].apply(lambda x:0 if x==0 else 2)
    # df1['label'] = df1['label'].apply(lambda x:0 if x==0 else 2)
    #
    # df_train = pd.concat([df1,df2],axis=0)
    # df_val = pd.concat([df3,df4],axis=0)
    #
    # df_train.to_csv('/Users/maoyufeng/Downloads/违规数据集/train.csv',index=False,encoding='utf-8')
    # df_val.to_csv('/Users/maoyufeng/Downloads/违规数据集/val.csv',index=False,encoding='utf-8')
    # print(1)