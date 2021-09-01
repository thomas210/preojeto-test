import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Teste")
#page_icon="images/val_temp.png"

paginas = ["Home", "leitura", "inputs", "Machine Learning", "Colunas"]

selecao = st.sidebar.selectbox(
    "Escolha a página",
    paginas
)

if selecao == "Home":

    st.write("# Header - Projeto de teste para utilizacao do streamlit")

    st.write("## subheader - Projeto de teste para utilizacao do streamlit")

    st.code("streamlit run teste.py")

elif selecao == "leitura":

    st.write("É possível ler dados de dataframe e visualizar no streamlit")

    df = pd.read_csv("baseCompleta-1617664035.csv", sep=";")

    st.dataframe(df)

    chart_data = pd.DataFrame(np.random.randn(20,3),columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

    st.area_chart(chart_data)

    st.bar_chart(chart_data)

elif selecao == "inputs":

    #with st.form("form"):

    check = st.checkbox('Checkbox')

    ratio = st.radio("RadioButton", [1,2,3])

    select = st.selectbox("Select", [1,2,3])

    multi = st.multiselect("MultiSelect", [1,2,3])

    slider = st.slider("Slide", min_value=4, max_value=24)

    texto = st.text_input("Campo de texto", value="padrao")

    numero = st.number_input("numero", min_value=6)

    res = {
        "check": check,
        "ratio": ratio,
        "select": select,
        "multi": multi,
        "slider": slider,
        "texto": texto,
        "numero": numero
    }

    st.write(res)

elif selecao == "Machine Learning":

    st.write("## Preencha os campos com as informações para realizar o diagnóstico:")

    with st.form("diagnostico"):

        DIAS = st.number_input("Quantos dias está sentindo os sintomas?", min_value=0, format="%d")

        st.write ("Informe os sintomas:")

        FEBRE =st.checkbox("Febre")

        MIALGIA =st.checkbox("Mialgia", help="Dor muscular")

        CEFALEIA =st.checkbox("Cefaleia", help="Dor de cabeça")

        EXANTEMA =st.checkbox("Exantema", help="Manchas vermelhas em um região")

        NAUSEA =st.checkbox("Náusea")

        DOR_COSTAS =st.checkbox("Dor nas costas")

        CONJUNTVIT =st.checkbox("Conjutivite")

        ARTRITE =st.checkbox("Artrite", help="Inflamação das articulações")

        ARTRALGIA =st.checkbox("Artralgia", help="Dor nas articulações")

        PETEQUIA_N =st.checkbox("Petéquias", help="Pequenas manchas vermelhas ou marrom que surgem geralmente aglomeradas, mais frequentemente nos braços, pernas ou barriga")

        DOR_RETRO =st.checkbox("Dor Retroorbital", help="Dor ao redor dos olhos")

        DIABETES =st.checkbox("Diabetes", help="")

        HIPERTENSA =st.checkbox("Hipertensão", help="")

        if (st.form_submit_button("Realizar Diagnóstico")):

            dados = []

            dados.append([
                FEBRE, MIALGIA, CEFALEIA,
                EXANTEMA, NAUSEA, DOR_COSTAS,
                CONJUNTVIT, ARTRITE, ARTRALGIA,
                PETEQUIA_N, DOR_RETRO,
                DIABETES, HIPERTENSA, DIAS
            ]) 

            with open('gradient_model1.pkl', 'rb') as f:
                model = pickle.load(f)
                doenca = model.predict(dados)[0]
                prob = model.predict_proba(dados)

            doencas = ["CHIKUNGUNYA", "DENGUE", "OUTRAS_DOENCAS"]
            doencas_texto = ["Chikungunya", "Dengue", "Inconclusivo"]

            
            for count, value in enumerate(doencas):
                if (value == doenca):
                    d = doencas_texto[count]
                    p = prob[0][count]

            st.write(f'O resultado do diagnóstico foi **{d}**')

            res_df = pd.DataFrame(prob, columns=doencas_texto, index=["Probabilidade"])

            df_style = res_df.style.format(
                {'Dengue':'{:.2%}',
                'Chikungunya':'{:.2%}',
                'Inconclusivo':'{:.2%}'}
            )


            st.write("Abaixo é possível obersar o resultados detalhado do diagnóstico:")
            st.dataframe(df_style)

            st.write("**AVISO: Este diagnóstico não substitui a avaliação médica, procure a unidade de saúde mais próxima!**")

            st.balloons()

elif (selecao == 'Colunas'):

    st.write("## Colunas")

    col_1, dotlab_col, col_3 = st.columns(3)

    col_1.image("DotLab.png")
    dotlab_col.image("DotLab.png")
    col_3.image("DotLab.png")

    st.image("DotLab.png")
