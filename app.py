#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : 7.demo_app.py
# @Author: nixin
# @Date  : 2021/11/27



from PIL import Image
import time
import datetime as datetime
from scipy import spatial
from gensim.models import word2vec
from keras.models import load_model
from LSTM.config import siamese_config
from LSTM.inputHandler import create_test_data, word_embed_meta_data
from simpletransformers.question_answering import QuestionAnsweringModel
from functools import partial
from functions import *
from skcriteria import Data, MAX, MIN
from skcriteria.madm import simple, closeness




#===================#
# Streamlit code
#===================#

# st.title('PatentSolver')
st.markdown("<h1 style='text-align: center; color: orange;'>PatentSolver</h1>", unsafe_allow_html=True)
image = Image.open('profile.png')
col1,mid, col2 = st.columns([50,10,30])
with col1:
    st.header('Achieve inventive ideas from U.S. Patents.')
with col2:
    st.image(image, width=150)

st.write('🚀 This demo app aims to explore latent inventive solutions from different domain U.S. patents.')
st.write('🎈 Click on top left corner button ➡️ to start.')
st.caption('🤖️ According to natural language processing-related techniques associated with semantic similarity computation, question answering system, and multiple criteria decision analysis,'
           ' this demo app is finally here.')
st.caption('📼 Introduction video: https://youtu.be/asDsOCuFprQ')
st.caption('📧 Please play it and send us feedback (nxnixin at gmail.com) since it is still very young :)')


add_selectbox = st.sidebar.selectbox(
    "Which function would you like to choose?",
    ('Start from the following options',"1. Patent details scraper", "2. Prepare patents (.txt) ", "3. Extract problems from patents", "4. Similar problem extractor", "5. Problem-solution matching", "6. Inventive solutions ranking")
)

#===================#
# Function 1
#===================#

if add_selectbox == '1. Patent details scraper':
    # st.title('PatentSolver_patent details')

    app_target = "To scrape details of the given U.S. patents"

    st.subheader(app_target)

    # user types the inputs
    user_input_patent_number = st.text_input('Type patent number')
    st.caption('1. use "," to separate if many. 2. please delete previous inputs '
               'when change or add new patents. 3. E.g. US20050210008A1, US9533047. '
               '4. Google patent search web: https://patents.google.com/')



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~ prepare patents ~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    if st.button('Run'):
        with st.spinner('Wait for it...'):
            start_time = time.time()

            list_of_patents = patentinput( user_input_patent_number)



            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~ Parameters for data_patent_details file ~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            path_to_data = "data_patent_details/"  #### don't forget to change


            ## Create csv file to store the data_patent_details from the patent runs
            #  (1) Specify column order of patents
            #  (2) Create csv if it does not exist in the data_patent_details path
            data_column_order = ['inventor_name',
                                'assignee_name_orig',
                                'assignee_name_current',
                                'pub_date',
                                'priority_date',
                                'grant_date',
                                'filing_date',
                                'forward_cite_no_family',
                                'forward_cite_yes_family',
                                'backward_cite_no_family',
                                'backward_cite_yes_family',
                                'patent',
                                'url',
                                 'abstract_text']

            if 'edison_patents.csv' in os.listdir(path_to_data):
                os.remove( path_to_data + 'edison_patents.csv')  # delete previous csv file
                with open(path_to_data + 'edison_patents.csv','w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data_column_order)
            else:
                with open(path_to_data + 'edison_patents.csv','w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data_column_order)
            #
            #
            ########### Run pool process #############
            if __name__ == "__main__":

                ## Create lock to prevent collisions when processes try to write on same file
                l = mp.Lock()

                ## Use a pool of workers where the number of processes is equal to
                ##   the number of cpus - 1
                with poolcontext(processes=mp.cpu_count()-1,initializer=init,initargs=(l,)) as pool:
                    pool.map(partial(single_process_scraper,path_to_data_file=path_to_data + 'edison_patents.csv',
                                                            data_column_order=data_column_order),
                                                            list_of_patents)



            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~ clean raw data_patent_details ~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            ##read Google scrawer's results
            table = pd.read_csv('data_patent_details/edison_patents.csv')

            # clean raw patent results
            results = clean_patent(table)



            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~ count number ~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



            results = count_patent(results)
            st.success('Done!')
            st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))

            # function of running
            # if st.button('Run'):
            st.dataframe(results)

            csv = convert_df(results)  # to download results
            st.download_button(
                label="Download",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )


#===================#
# Function 2
#===================#

elif add_selectbox == '2. Prepare patents (.txt) ':

    file_path_saved = 'patent_text/'

    app_target = "To convert patents (.xml) file to patents (.txt) file"

    st.subheader(app_target)

    st.caption(
        '🚥 Please firstly choose "Patent Grant Full Text Data (No Images)" from https://developer.uspto.gov/product/patent-grant-full-text-dataxml to download U.S. patents (.xml) you want.')

    uploaded_files = st.file_uploader("Choose U.S. patent files", type='XML', accept_multiple_files=True)

    if st.button('run'):
        with st.spinner('Wait for it...'):
            start_time = time.time()

            path = os.listdir('patent_text/')

            if len(path) == 0:
                print("Directory is empty")

                for uploaded_file in uploaded_files:
                    XMLtoTEXT(patent_xml=uploaded_file, saved_file_path=file_path_saved)


            else:
                print("Directory is not empty")
                files = glob.glob('patent_text/*')
                for each in files:
                    os.remove(each)  # remove previous files


                for uploaded_file in uploaded_files:
                    XMLtoTEXT(patent_xml=uploaded_file, saved_file_path=file_path_saved)




            path = os.listdir('patent_text/')

            st.write(path)
            st.success('Done!')
            st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))



            # download patents (txt) by zip file
            create_download_zip(zip_directory='patent_text',
                                zip_path='zip_file/',
                                filename='US_patents')


#===================#
# Function 3
#===================#
elif add_selectbox == '3. Extract problems from patents':


    app_target = "To extract problems from patents"

    st.subheader(app_target)

    st.caption('🚨 Please choose one or several patents (from Function 2).')

    uploaded_files = st.file_uploader("Choose U.S. patents", type='txt', accept_multiple_files=True)
    print(uploaded_files)

    # check the folder is empty or not
    if len(os.listdir('Data/input/US_patents')) == 0:
        print("Directory is empty")
        # save uploaded files into the folder(//input/US_patents)
        for f in uploaded_files:
            if uploaded_files is not None:
                save_uploadedfile(f)
    else:
        print("Directory is not empty")
        files = glob.glob('Data/input/US_patents/*')
        for each in files:
            os.remove(each) #remove previous files
        # save uploaded files into the folder(//input/US_patents)
        for f in uploaded_files:
            if uploaded_files is not None:
                save_uploadedfile(f)

    if st.button('Extract'):
        with st.spinner('Wait for it...'):
            start_time = time.time()
            extractor('US_patents') #extract problems from this folder (//US_patents)
            st.success('Done!')
            st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))


        table = extract_info_text()
        st.dataframe(table)

        csv = convert_df(table) #to download problem results
        st.download_button(
            label="Download",
            data = csv,
            file_name = 'results.csv',
            mime = 'text/csv',
        )

# ===================#
# Function 4
# ===================#
elif add_selectbox == '4. Similar problem extractor':

    app_target = "To extract similar problems from different domains U.S. patents"

    st.subheader(app_target)

    st.caption('👨‍💻 Please type one target problem you want from Function 3.')

    # user types the inputs
    user_input_patent_sentence = st.text_input('Type one patent problem sentence')


    # choose patent domain
    select_domain = st.selectbox('Which domain it belongs to?',
                                        ['A (Human necessities)', 'B (Performing operations; transporting)', 'C (Chemistry; metallurgy)','D (Textiles; paper)', 'E (Fixed constructions)', 'F (Mechanical engineering; lighting; heating; weapons; blasting engines or pumps','G (Physics)',' H (Electricity)'])
    user_input_domain = input_domain(select_domain) #get domain lable like A B C



    # choose one of trained models
    select_model = st.selectbox('Which model do you want?',
                                        ['IDM-Similar', 'SAM-IDM'])
    st.caption('1. ⚙️ IDM-Similar based on Word2vec neural networks \n 2. ⚙️ SAM-IDM based on LSTM neural networks')



    # the function of choosing time period for comparied problems
    choose_time_range = st.date_input("Time Period", [datetime.date(2019, 5, 1), datetime.date(2019, 5, 31)])
    start = datetime.datetime.combine(choose_time_range[0], datetime.datetime.min.time()) #recevie the input of start time
    end = datetime.datetime.combine(choose_time_range[1], datetime.datetime.min.time()) #recevie the input of end time
    st.caption('1. 🥱 The longer time period will result in the longer waiting time. Suggest one month. \n '
               '2. 🗓 The problem sample corpus is from 2006-2020 year, please choose among this period. ')

    start_year = int(start.strftime("%Y"))
    start_month = int(start.strftime("%m"))

    end_year = int(end.strftime("%Y"))
    end_month = int(end.strftime("%m"))

    if select_model== 'IDM-Similar':
        select_threshold = st.slider('Similarity Threshold:', 0.6, 1.0, 0.8)
    else:
        select_threshold = st.slider('Similarity Threshold:', 0.1, 1.0, 0.2)


    if select_model == 'IDM-Similar': #user chooses IDM-Similar
        if st.button('Run'):
            with st.spinner('Wait for it...'):
                start_time = time.time()


        ################################
        # IDM-Similar model (Word2vec)
        ################################

                # load the trained word vector model
                model = word2vec.Word2Vec.load('Word2vec/trained_word2vec.model')
                index2word_set = set(model.wv.index2word)


                #read problem patent corpus
                problem_corpus = pd.read_csv('data_problem_corpus/problem_corpus_full_cleaned.csv')
                # problem_corpus = problem_corpus.head(500)

                print('--------------------')
                print(problem_corpus.columns)
                print('--------------------')

                target_problem = user_input_patent_sentence
                target_domain = user_input_domain

                # remove the same domain's problems
                problem_corpus = problem_corpus[problem_corpus.Domain != target_domain]

                # choose the month period
                problem_corpus = choosing_month_period(problem_corpus = problem_corpus, start_year = start_year,
                                  end_year = end_year, start_month = start_month, end_month = end_month)

                print(problem_corpus)
                print(problem_corpus.columns)
                print('=======')

                # compute the similarity value
                value_1=[]
                for each_problem in problem_corpus['First part Contradiction']:
                    s1_afv = avg_feature_vector(target_problem, model=model, num_features=100, index2word_set=index2word_set)
                    s2_afv = avg_feature_vector(each_problem, model=model, num_features=100, index2word_set=index2word_set)
                    sim_value = format( 1 - spatial.distance.cosine(s1_afv, s2_afv), '.2f')
                    value_1.append(sim_value)

                print("++++++++++")
                problem_corpus[['similarity_value_1', 'target_problem']] = value_1, target_problem

                value_2=[]
                for each_problem in problem_corpus['Second part Contradiction']:
                    s1_afv = avg_feature_vector(target_problem, model=model, num_features=100, index2word_set=index2word_set)
                    s2_afv = avg_feature_vector(each_problem, model=model, num_features=100, index2word_set=index2word_set)
                    sim_value = format( 1 - spatial.distance.cosine(s1_afv, s2_afv), '.2f')
                    value_2.append(sim_value)
                problem_corpus['similarity_value_2'] = value_2

                print("++++++++++")
                print(problem_corpus)
                print(problem_corpus.columns)
                print("++++++++++")

                problem_corpus_1 = problem_corpus[['patent_number', 'Domain', 'First part Contradiction', 'publication_date', 'publication_year','publication_month', 'label', 'similarity_value_1', 'target_problem']]
                problem_corpus_1 = problem_corpus_1.rename(columns = {'First part Contradiction': 'problem', 'similarity_value_1' : 'similarity_value'})

                problem_corpus_2 = problem_corpus[
                    ['patent_number', 'Domain', 'Second part Contradiction', 'publication_date', 'publication_year', 'publication_month', 'label',
                     'similarity_value_2', 'target_problem']]
                problem_corpus_2 = problem_corpus_2.rename(columns={'Second part Contradiction': 'problem', 'similarity_value_2' : 'similarity_value'})

                problem_corpus_final = pd.concat([problem_corpus_1, problem_corpus_2], ignore_index=True, sort=False)

                print(problem_corpus_final)
                print(problem_corpus_final.columns)
                print(type(select_threshold))
                print(select_threshold)
                problem_corpus_final.to_csv('result_test.csv',index=False)
                print('=================')


                # choose the resutls that are bigger than the similarity threshold
                problem_corpus_final = problem_corpus_final[problem_corpus_final['similarity_value'].astype(str)>= str(select_threshold)]
                problem_corpus_final= problem_corpus_final[['patent_number', 'Domain','problem', 'similarity_value', 'target_problem']]


                # dropping duplicate values
                problem_corpus_final = problem_corpus_final.drop_duplicates(ignore_index=True)


                problem_corpus_final.to_csv('Word2vec/simialrity_result/test.csv', index=False)
                print(problem_corpus_final)

                st.success('Done!')
                st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))

                # show results
                st.dataframe(problem_corpus_final)

                csv = convert_df(problem_corpus_final)  # to download results
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )
        # ==================
    else: #select_model == 'SAM-IDM':
        if st.button('Run'):
            with st.spinner('Wait for it...'):
                start_time = time.time()


        ################################
        # SAM-IDM model (LSTM)
        ################################


                df = pd.read_csv('LSTM/sample_data.csv')
                print(df.head())


                sentences1 = list(df['sentences1'])
                sentences2 = list(df['sentences2'])

                tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2, siamese_config['EMBEDDING_DIM'])

                model = load_model(
                    "LSTM/choosed_checkpoit/lstm_50_50_0.17_0.25.h5",
                    None, False)

                problem_corpus = pd.read_csv(
                    'data_problem_corpus/problem_corpus_full_cleaned.csv')

                target_problem = user_input_patent_sentence
                target_domain = user_input_domain

                # remove the same domain's problems
                problem_corpus = problem_corpus[problem_corpus.Domain != target_domain]

                # choose the month period
                problem_corpus = choosing_month_period(problem_corpus=problem_corpus, start_year=start_year,
                                                       end_year=end_year, start_month=start_month, end_month=end_month)


                problem_corpus.reset_index(drop=True, inplace=True) # reset the index of the dataframe(must do this step)


                print(problem_corpus)
                print(problem_corpus.columns)
                print('=======')


                # read specific column
                column1 = problem_corpus['First part Contradiction']
                print(type(column1))
                print(column1.head())
                print('++++++++++++++++')

                for i in range(0, len(problem_corpus)):
                    ss1 = column1[i]
                    ss2 = target_problem

                    test_sentence_pairs = [(ss1, ss2)]
                    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,
                                                                              siamese_config['MAX_SEQUENCE_LENGTH'])

                    pred = model.predict([test_data_x1, test_data_x2, leaks_test], batch_size=1000, verbose=2).ravel()

                    problem_corpus.loc[i, 'similarity_value_1'] = pred
                # ==========

                column2 = problem_corpus['Second part Contradiction']
                for i in range(0, len(problem_corpus)):
                    ss1 = column2[i]
                    ss2 = target_problem

                    test_sentence_pairs = [(ss1, ss2)]
                    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,
                                                                              siamese_config['MAX_SEQUENCE_LENGTH'])

                    pred = model.predict([test_data_x1, test_data_x2, leaks_test], batch_size=1000, verbose=2).ravel()

                    problem_corpus.loc[i, 'similarity_value_2'] = pred

                problem_corpus['target_problem'] = target_problem

                problem_corpus = problem_corpus.round({'similarity_value_1': 2, 'similarity_value_2': 2})  # save 4 digits after point
                print(problem_corpus.head())
                print(problem_corpus.columns)


                problem_corpus_1 = problem_corpus[['patent_number', 'Domain', 'First part Contradiction', 'publication_date', 'publication_year','publication_month', 'label', 'similarity_value_1', 'target_problem']]
                problem_corpus_1 = problem_corpus_1.rename(columns = {'First part Contradiction': 'problem', 'similarity_value_1' : 'similarity_value'})

                problem_corpus_2 = problem_corpus[
                    ['patent_number', 'Domain', 'Second part Contradiction', 'publication_date', 'publication_year', 'publication_month', 'label',
                     'similarity_value_2', 'target_problem']]
                problem_corpus_2 = problem_corpus_2.rename(columns={'Second part Contradiction': 'problem', 'similarity_value_2' : 'similarity_value'})

                problem_corpus_final = pd.concat([problem_corpus_1, problem_corpus_2], ignore_index=True, sort=False)

                print(problem_corpus_final)
                print(problem_corpus_final.columns)
                print(type(select_threshold))
                print(select_threshold)
                print('=================')


                # choose the resutls that are bigger than the similarity threshold

                problem_corpus_final = problem_corpus_final[problem_corpus_final['similarity_value']>= select_threshold]
                problem_corpus_final= problem_corpus_final[['patent_number', 'Domain','problem', 'similarity_value', 'target_problem']]


                # dropping duplicate values
                problem_corpus_final = problem_corpus_final.drop_duplicates(ignore_index=True)


                print(problem_corpus_final)

                st.success('Done!')
                st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))

                # show results
                st.dataframe(problem_corpus_final)

                csv = convert_df(problem_corpus_final)  # to download results
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )



    # future function: add function of providing own dataset

# ===================#
# Function 5
# ===================#

if add_selectbox == '5. Problem-solution matching':
    # st.title('PatentSolver_inventive solution matching')

    app_target = "To provide latent inventive solutions for the target problem"
    st.subheader(app_target)

    st.caption('⌨️‍ Please use similar problem results from Function 4. ')
    st.caption('🚁 IDM-Matching model behind here is based on XLNet neural networks.')


    uploaded_file = st.file_uploader("upload your similar problem file", type='csv')

    if uploaded_file is not None:
        # choose GPU
        select_GPU = st.selectbox('Do you have GPU(s)?',
                                  ['No', 'Yes'])
        st.caption('1. 💰 We don\'t provide GPU since the cost. \n 2. 🎢 Please choose Yes when you run it on your own '
                   'GPU and it will greatly accelerate the process.')

        if select_GPU == 'No':
            use_cuda = "False"
        else:
            use_cuda = "True"

        if st.button('Run'):
            with st.spinner('Wait for it...'):
                start_time = time.time()

                data = pd.read_csv(uploaded_file)
                data = creat_query_id(data)
                context_infor = pd.read_csv(
                    'data_problem_corpus/problem_corpus_full_cleaned.csv')

                context_infor = context_infor[['patent_number', 'Context']]
                # get context table
                final_context = pd.merge(data, context_infor, on=['patent_number'])
                final_context.to_csv(
                    'data_context/context_information.csv',
                    index=False)
                print('++++++++++++')
                print(final_context.head())
                print(final_context.columns)

                csv_file = 'data_context/context_information.csv'
                json_file = 'data_context/context_information.json'
                csv_to_json(csv_file, json_file)  # convert context.csv to context.json

                prediction_file = 'data_context/context_information.json'
                prediction_output = 'data_context/QA_result.json'

                model = QuestionAnsweringModel('xlnet', 'trained_xlnet_model',
                                               use_cuda=False)  # when don't have GPU, choose use_cuda=False
                QA_prediction(prediction_file, prediction_output, model)  # predict solutions by QA system

                input_file = 'data_context/QA_result.json'
                output_file = 'data_context/QA_result.csv'

                json_to_csv(input_file, output_file)

                similarity_result = pd.read_csv(
                    'data_context/context_information.csv')

                id_result = pd.read_csv(
                    'data_context/QA_result.csv')

                final_result = similarity_result.merge(id_result, on=['id'], how='left')
                print(final_result.head())
                final_result = final_result[
                    ['target_problem', 'problem', 'similarity_value', 'patent_number', 'Domain', 'answer']]
                final_result = final_result.rename(
                    columns={'problem': 'similar_problem', 'answer': 'latent_inventive_solutions'})
                final_result.to_csv(
                    'data_context/QA_result_final.csv',
                    index=False)
                st.dataframe(final_result)

                csv = convert_df(final_result)  # to download solution results
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )

            st.success('Done!')
            st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))

# ===================#
# Function 6
# ===================#
if add_selectbox == '6. Inventive solutions ranking':

    # st.title('PatentSolver_rank latent inventive solutions')

    app_target = "To rank latent inventive solutions"
    st.subheader(app_target)

    st.caption('⌨️‍ Please use similar problem results from Function 5. ')
    st.caption('🙇‍ ️PatRIS model behind here is based on the multiple criteria decision analysis approach named TOPSIS.')

    uploaded_file = st.file_uploader("upload your problem-solution file", type='csv')

    if uploaded_file is not None:
        if st.button('Run'):
            st.write('Weight assignments:')

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric('IN', '0.1')
            col2.metric('FCNF', '0.3')
            col3.metric('FCYF', '0.1')
            col4.metric('BCNF', '0.1')
            col5.metric('BCYF', '0.1')
            col6.metric('SV', '0.3')

            with st.expander('See explanation'):
                st.write('Inventive solutions ranking features: \n'
                         'IN (inventor_name): the number of inventors involved in the patent.\n'
                         'FCNF (forward_cite_no_family): Forward Citations that are not family-to-family cites.\n'
                         'FCYF (forward_cite_yes_family): Forward Citations that are family-to-family cites.\n'
                         'BCNF (backward_cite_no_family): Backward Citations that are not family-to-family cites.\n'
                         'BCYF (backward_cite_yes_family): Backward Citations that are family-to-family cites.\n'
                         'SV (similarity_value): similarity value between similar pairwise problems.\n')

            with st.spinner('Wait for it...'):
                start_time = time.time()

                df = pd.read_csv(uploaded_file)
                print(df.columns)

                patent_number = []
                for patent in df['patent_number']:  # take patent numbers
                    patent_number.append(patent)

                print(patent_number)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                # ~~~ Parameters for data_patent_details file ~~~ #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                path_to_data = "MCDA/data/"  #### don't forget to change

                ## Create csv file to store the data_patent_details from the patent runs
                #  (1) Specify column order of patents
                #  (2) Create csv if it does not exist in the data_patent_details path
                data_column_order = ['inventor_name',
                                     'assignee_name_orig',
                                     'assignee_name_current',
                                     'pub_date',
                                     'priority_date',
                                     'grant_date',
                                     'filing_date',
                                     'forward_cite_no_family',
                                     'forward_cite_yes_family',
                                     'backward_cite_no_family',
                                     'backward_cite_yes_family',
                                     'patent',
                                     'url',
                                     'abstract_text']

                if 'edison_patents.csv' in os.listdir(path_to_data):
                    os.remove(path_to_data + 'edison_patents.csv')  # delete previous csv file
                    with open(path_to_data + 'edison_patents.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data_column_order)
                else:
                    with open(path_to_data + 'edison_patents.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data_column_order)
                #
                #
                ########### Run pool process #############
                if __name__ == "__main__":
                    ## Create lock to prevent collisions when processes try to write on same file
                    l = mp.Lock()

                    ## Use a pool of workers where the number of processes is equal to
                    ##   the number of cpus - 1
                    with poolcontext(processes=mp.cpu_count() - 1, initializer=init, initargs=(l,)) as pool:
                        pool.map(partial(single_process_scraper, path_to_data_file=path_to_data + 'edison_patents.csv',
                                         data_column_order=data_column_order),
                                 patent_number)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                # ~~~ clean raw data_patent_details ~~~ #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

                ##read Google scrawer's results
                table = pd.read_csv(
                    'MCDA/data/edison_patents.csv')

                # clean raw patent results
                results = clean_patent(table)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                # ~~~ count number ~~~ #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

                results = count_patent(results)
                print(results.columns)
                results.to_csv(
                    'MCDA/data/cleaned_count_patents.csv',
                    index=False)
                results_show = results[['patent_number', 'inventor_name', 'count_inventor_name',
                                        'assignee_name_orig', 'count_assignee_name', 'assignee_name_current',
                                        'count_assignee_name_current', 'forward_cite_no_family',
                                        'count_forward_cite_no_family', 'forward_cite_yes_family',
                                        'count_forward_cite_yes_family', 'backward_cite_no_family',
                                        'count_backward_cite_no_family', 'backward_cite_yes_family',
                                        'count_backward_cite_yes_family']]
                st.write('Related patent details:')
                st.dataframe(results_show)  # show patent count details

                print(len(df))
                print('==========')
                # clean null soltuions
                solutions = df[df['latent_inventive_solutions'] != '[]']
                print(len(solutions))

                count = results_show[['patent_number', 'count_inventor_name', 'count_forward_cite_no_family',
                                      'count_forward_cite_yes_family', 'count_backward_cite_no_family',
                                      'count_backward_cite_yes_family']]

                count = pd.merge(count, solutions[['patent_number', 'similarity_value']], on='patent_number')
                st.write('Solutions ranking criteria:')
                st.dataframe(count)  # show ranking criteria details

                print('=======')
                print(count.columns)

                ## project the goodness for each column
                criteria_data = Data(count.iloc[:, 1:7], [MAX, MAX, MAX, MAX, MAX, MAX],
                                     anames=count['patent_number'],
                                     cnames=count.columns[1:7],
                                     weights=[0.1, 0.3, 0.1, 0.1, 0.1, 0.3])  ##assign weights to attributes
                print(criteria_data)
                print('++++++++')

                print('==========')
                dm = closeness.TOPSIS(
                    mnorm="sum")  # change the normalization criteria of the alternative matric to sum (divide every value by the sum opf their criteria)
                dec = dm.decide(criteria_data)
                print(dec)
                print("Ideal:", dec.e_.ideal)
                print("Anti-Ideal:", dec.e_.anti_ideal)
                print("Closeness:", dec.e_.closeness)  ##print each rank's value

                count['rank_topsis'] = dec.e_.closeness
                count = count.sort_values(by='rank_topsis', ascending=False)
                print(count.columns)
                print(count)
                print(len(count))

                rank = []
                for i in range(len(count)):
                    i = i + 1
                    rank.append(i)
                print(rank)

                count['rank'] = rank
                print(count)
                print(count.columns)
                count = count[['rank', 'patent_number', 'count_inventor_name', 'count_forward_cite_no_family',
                               'count_forward_cite_yes_family', 'count_backward_cite_no_family',
                               'count_backward_cite_yes_family', 'similarity_value']]

                final = pd.merge(count, df, on=('patent_number', 'similarity_value'))
                final = final[
                    ['target_problem', 'latent_inventive_solutions', 'rank', 'similar_problem', 'similarity_value',
                     'Domain', 'patent_number', 'count_inventor_name',
                     'count_forward_cite_no_family', 'count_forward_cite_yes_family',
                     'count_backward_cite_no_family', 'count_backward_cite_yes_family']]
                print('+++++')
                print(final.columns)

                st.write('Inventive solutions ranking results according to TOPSIS:')
                st.dataframe(final)

                st.success('Done!')
                st.write("Process is finished within %s seconds" % round(time.time() - start_time, 2))

                csv = convert_df(final)  # to download solution results
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )
