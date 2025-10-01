import os
import re
import pandas as pd

# ----------------------------
# 1. DEFINIÇÃO DE CAMINHOS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "state_of_data.csv")
STAGING_PATH = os.path.join(BASE_DIR, "data", "staging", "state_of_data_clean.csv")

# ----------------------------
# 2. FUNÇÕES
# ----------------------------
def carregar_dados(df):
    df = pd.read_csv(df, encoding="utf-8")
    df.columns = df.columns.astype(str)
    return df

def limpar_dados(df):
    df = df.dropna(axis=1, how="all")

    def nome_limpo(col):
        if "_" in col:
            return col.split("_", 1)[1]
        return col

    df.columns = [nome_limpo(c) for c in df.columns]

    df.columns = (
        pd.Series(df.columns)
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .tolist()
    )

    for col in df.select_dtypes(include="object"):
        try:
            df[col] = df[col].str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    if 'idade' in df.columns:
        df['idade'] = pd.to_numeric(df['idade'], errors="coerce")

    if 'data/hora_envio' in df.columns:
        df['data/hora_envio'] = pd.to_datetime(df['data/hora_envio'], dayfirst=True, errors='coerce').dt.date

    '''colunas_numericas = [
        "remuneração/salário",
        "benefícios",
        "propósito_do_trabalho_e_da_empresa",
        "flexibilidade_de_trabalho_remoto",
        "ambiente_e_clima_de_trabalho",
        "oportunidade_de_aprendizado_e_trabalhar_com_referências",
        "oportunidades_de_crescimento",
        "maturidade_da_empresa_em_termos_de_tecnologia_e_dados",
        "relação_com_os_gestores_e_líderes",
        "reputação_que_a_empresa_tem_no_mercado",
        "gostaria_de_trabalhar_em_outra_área"
    ]

    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")'''

    df = df.drop_duplicates()

    colunas_para_remover = [
        "não_acredito_que_minha_experiência_profissional_seja_afetada",
        "sim,_devido_a_minha_cor/raça/etnia",
        "sim,_devido_a_minha_identidade_de_gênero",
        "sim,_devido_ao_fato_de_ser_pcd",
        "estudos_ad_hoc_com_o_objetivo_de_confirmar_hipóteses,_realizar_modelos_preditivos,_forecasts,_análise_de_cluster_para_resolver_problemas_pontuais_e_responder_perguntas_das_áreas_de_negócio.",
        "coletando_e_limpando_dos_dados_que_uso_para_análise_e_modelagem.",
        "entrando_em_contato_com_os_times_de_negócio_para_definição_do_problema,_identificar_a_solução_e_apresentação_de_resultados.",
        "desenvolvendo_modelos_de_machine_learning_com_o_objetivo_de_colocar_em_produção_em_sistemas_(produtos_de_dados).",
        "colocando_modelos_em_produção,_criando_os_pipelines_de_dados,_apis_de_consumo_e_monitoramento.",
        "cuidando_da_manutenção_de_modelos_de_machine_learning_já_em_produção,_atuando_no_monitoramento,_ajustes_e_refatoração_quando_necessário.",
        "realizando_construções_de_dashboards_em_ferramentas_de_bi_como_powerbi,_tableau,_looker,_qlik,_etc.",
        "utilizando_ferramentas_avançadas_de_estatística_como_sas,_spss,_stata_etc,_para_realizar_análises.",
        "criando_e_dando_manutenção_em_etls,_dags_e_automações_de_pipelines_de_dados.",
        "criando_e_gerenciando_soluções_de_feature_store_e_cultura_de_mlops.",
        "criando_e_mantendo_a_infra_que_meus_modelos_e_soluções_rodam_(clusters,_servidores,_api,_containers,_etc.)",
        "treinando_e_aplicando_llm's_para_solucionar_problemas_de_negócio.",
        "ferramentas_de_bi_(powerbi,_looker,_tableau,_qlik_etc).",
        "planilhas_(excel,_google_sheets_etc).",
        "ambientes_de_desenvolvimento_local_(r_studio,_jupyterlab,_anaconda).",
        "ambientes_de_desenvolvimento_na_nuvem_(google_colab,_aws_sagemaker,_kaggle_notebooks_etc).",
        "ferramentas_de_automl_(datarobot,_h2o,_auto_keras_etc).",
        "ferramentas_de_etl_(apache_airflow,_nifi,_stitch,_fivetran,_pentaho_etc).",
        "plataformas_de_machine_learning_(tensorflow,_azure_machine_learning,_kubeflow_etc).",
        "feature_store_(feast,_hopsworks,_aws_feature_store,_databricks_feature_store_etc).",
        "sistemas_de_controle_de_versão_(github,_dvc,_neptune,_gitlab_etc).",
        "plataformas_de_data_apps_(streamlit,_shiny,_plotly_dash_etc).",
        "ferramentas_de_estatística_avançada_como_spss,_sas,_etc.",
        "utilizo_modelos_de_regressão_(linear,_logística,_glm).",
        "utilizo_redes_neurais_ou_modelos_baseados_em_árvore_para_criar_modelos_de_classificação.",
        "desenvolvo_sistemas_de_recomendação_(recsys).",
        "utilizo_métodos_estatísticos_bayesianos_para_analisar_dados.",
        "utilizo_técnicas_de_nlp_(natural_language_processing)_para_análisar_dados_não_estruturados.",
        "utilizo_métodos_estatísticos_clássicos_(testes_de_hipótese,_análise_multivariada,_sobrevivência,_dados_longitudinais,_inferência_estatistica)_para_analisar_dados.",
        "utilizo_cadeias_de_markov_ou_hmm\s_para_realizar_análises_de_dados.",
        "desenvolvo_técnicas_de_clusterização_(k_means,_spectral,_dbscan_etc).",
        "realizo_previsões_através_de_modelos_de_séries_temporais_(time_series).",
        "utilizo_modelos_de_reinforcement_learning_(aprendizado_por_reforço).",
        "utilizo_modelos_de_machine_learning_para_detecção_de_fraude.",
        "utilizo_métodos_de_visão_computacional.",
        "utilizo_modelos_de_detecção_de_churn.",
        "utilizo_llm's_para_solucionar_problemas_de_negócio.",
        "sou_responsável_pela_coleta_e_limpeza_dos_dados_que_uso_para_análise_e_modelagem.",
        "sou_responsável_por_entrar_em_contato_com_os_times_de_negócio_para_definição_do_problema,_identificar_a_solução_e_apresentação_de_resultados.",
        "desenvolvo_modelos_de_machine_learning_com_o_objetivo_de_colocar_em_produção_em_sistemas_(produtos_de_dados).",
        "sou_responsável_por_colocar_modelos_em_produção,_criar_os_pipelines_de_dados,_apis_de_consumo_e_monitoramento.",
        "cuido_da_manutenção_de_modelos_de_machine_learning_já_em_produção,_atuando_no_monitoramento,_ajustes_e_refatoração_quando_necessário.",
        "realizo_construções_de_dashboards_em_ferramentas_de_bi_como_powerbi,_tableau,_looker,_qlik,_etc",
        "utilizo_ferramentas_avançadas_de_estatística_como_sas,_spss,_stata_etc,_para_realizar_análises.",
        "crio_e_dou_manutenção_em_etls,_dags_e_automações_de_pipelines_de_dados.",
        "crio_e_gerencio_soluções_de_feature_store_e_cultura_de_mlops.",
        "sou_responsável_por_criar_e_manter_a_infra_que_meus_modelos_e_soluções_rodam_(clusters,_servidores,_api,_containers,_etc.)",
        "treino_e_aplico_llm's_para_solucionar_problemas_de_negócio.",
        "processando_e_analisando_dados_utilizando_linguagens_de_programação_como_python,_r_etc.",
        "realizando_construções_de_dashboards_em_ferramentas_de_bi_como_powerbi,_tableau,_looker,_qlik_etc.",
        "criando_consultas_através_da_linguagem_sql_para_exportar_informações_e_compartilhar_com_as_áreas_de_negócio.",
        "utilizando_api's_para_extrair_dados_e_complementar_minhas_análises.",
        "realizando_experimentos_e_estudos_utilizando_metodologias_estatísticas_como_teste_de_hipótese,_modelos_de_regressão_etc.",
        "desenvolvendo/cuidando_da_manutenção_de_etl's_utilizando_tecnologias_como_talend,_pentaho,_airflow,_dataflow_etc.",
        "atuando_na_modelagem_dos_dados,_com_o_objetivo_de_criar_conjuntos_de_dados_como_data_warehouses,_data_marts,_datasets_etc.",
        "desenvolvendo/cuidando_da_manutenção_de_planilhas_para_atender_as_áreas_de_negócio.",
        "utilizando_ferramentas_avançadas_de_estatística_como_sas,_spss,_stata_etc,_para_realizar_análises_de_dados.",
        "nenhuma_das_opções_listadas_refletem_meu_dia_a_dia.",
        "scripts_python",
        "sql_&_stored_procedures",
        "apache_airflow",
        "apache_nifi",
        "luigi",
        "aws_glue",
        "talend",
        "pentaho",
        "alteryx",
        "stitch",
        "fivetran",
        "google_dataflow",
        "oracle_data_integrator",
        "ibm_datastage",
        "sap_bw_etl",
        "sql_server_integration_services_(ssis)",
        "sas_data_integration",
        "qlik_sense",
        "knime",
        "databricks",
        "não_utilizo_ferramentas_de_etl",
        "ferramentas_autonomia_area_de_negocios",
        "ferramentas_de_automl_como_h2o.ai,_data_robot,_bigml_etc.",
        "\"point_and_click\"_analytics_como_alteryx,_knime,_rapidminer_etc.",
        "product_metricts_&_insights_como_mixpanel,_amplitude,_adobe_analytics.",
        "ferramentas_de_análise_dentro_de_ferramentas_de_crm_como_salesforce_einstein_anaytics_ou_zendesk_dashboards.",
        "minha_empresa_não_utiliza_essas_ferramentas.",
        "não_sei_informar.",
        "processo_e_analiso_dados_utilizando_linguagens_de_programação_como_python,_r_etc.",
        "realizo_construções_de_dashboards_em_ferramentas_de_bi_como_powerbi,_tableau,_looker,_qlik_etc.",
        "crio_consultas_através_da_linguagem_sql_para_exportar_informações_e_compartilhar_com_as_áreas_de_negócio.",
        "utilizo_api\s_para_extrair_dados_e_complementar_minhas_análises.",
        "realizo_experimentos_e_estudos_utilizando_metodologias_estatísticas_como_teste_de_hipótese,_modelos_de_regressão_etc.",
        "desenvolvo/cuido_da_manutenção_de_etl\s_utilizando_tecnologias_como_talend,_pentaho,_airflow,_dataflow_etc.",
        "atuo_na_modelagem_dos_dados,_com_o_objetivo_de_criar_conjuntos_de_dados_como_data_warehouses,_data_marts_etc.",
        "desenvolvo/cuido_da_manutenção_de_planilhas_para_atender_as_áreas_de_negócio.",
        "utilizo_ferramentas_avançadas_de_estatística_como_sas,_spss,_stata_etc,_para_realizar_análises_de_dados."
        "desenvolvendo_pipelines_de_dados_utilizando_linguagens_de_programação_como_python,_scala,_java_etc.",
        "realizando_construções_de_etl\\s_em_ferramentas_como_pentaho,_talend,_dataflow_etc.",
        "atuando_na_integração_de_diferentes_fontes_de_dados_através_de_plataformas_proprietárias_como_stitch_data,_fivetran_etc.",
        "modelando_soluções_de_arquitetura_de_dados,_criando_componentes_de_ingestão_de_dados,_transformação_e_recuperação_da_informação.",
        "desenvolvendo/cuidando_da_manutenção_de_repositórios_de_dados_baseados_em_streaming_de_eventos_como_data_lakes_e_data_lakehouses.",
        "cuidando_da_qualidade_dos_dados,_metadados_e_dicionário_de_dados.",
        "possui_data_lake",
        "tecnologia_data_lake",
        "possui_data_warehouse",
        "tecnologia_data_warehouse",
        "ferramentas_de_qualidade_de_dados_(dia_a_dia)",
        "desenvolvo_pipelines_de_dados_utilizando_linguagens_de_programação_como_python,_scala,_java_etc.",
        "realizo_construções_de_etl's_em_ferramentas_como_pentaho,_talend,_dataflow_etc.",
        "atuo_na_integração_de_diferentes_fontes_de_dados_através_de_plataformas_proprietárias_como_stitch_data,_fivetran_etc.",
        "modelo_soluções_de_arquitetura_de_dados,_criando_componentes_de_ingestão_de_dados,_transformação_e_recuperação_da_informação.",
        "desenvolvo/cuido_da_manutenção_de_repositórios_de_dados_baseados_em_streaming_de_eventos_como_data_lakes_e_data_lakehouses.",
        "atuo_na_modelagem_dos_dados,_com_o_objetivo_de_criar_conjuntos_de_dados_como_data_warehouses,_data_marts,_datasets_etc.",
        "cuido_da_qualidade_dos_dados,_metadados_e_dicionário_de_dados.",
        "4.l.1_colaboradores_usando_ai_generativa_de_forma_independente_e_descentralizada",
        "4.l.2_direcionamento_centralizado_do_uso_de_ai_generativa",
        "4.l.3_desenvolvedores_utilizando_copilots",
        "4.l.4_ai_generativa_e_llms_para_melhorar_produtos_externos_para_os_clientes_finais",
        "4.l.5_ai_generativa_e_llms_para_melhorar_produtos_internos_para_os_colaboradores",
        "4.l.6_ia_generativa_e_llms_como_principal_frente_do_negócio",
        "4.l.7_ia_generativa_e_llms_não_é_prioridade",
        "4.l.8_não_sei_opinar_sobre_o_uso_de_ia_generativa_e_llms_na_empresa",
        "usa_chatgpt_ou_copilot_no_trabalho?",
        "4.m.1_não_uso_soluções_de_ai_generativa_com_foco_em_produtividade",
        "4.m.2_uso_soluções_gratuitas_de_ai_generativa_com_foco_em_produtividade",
        "4.m.3_uso_e_pago_pelas_soluções_de_ai_generativa_com_foco_em_produtividade",
        "4.m.4_a_empresa_que_trabalho_paga_pelas_soluções_de_ai_generativa_com_foco_em_produtividade",
        "4.m.5_uso_soluções_do_tipo_copilot",
        "microsoft_powerbi",
        "qlik_view/qlik_sense",
        "tableau",
        "metabase",
        "superset",
        "redash",
        "looker",
        "looker_studio(google_data_studio)",
        "amazon_quicksight",
        "sap_business_objects/sap_analytics",
        "oracle_business_intelligence",
        "salesforce/einstein_analytics",
        "sas_visual_analytics",
        "grafana",
        "fazemos_todas_as_análises_utilizando_apenas_excel_ou_planilhas_do_google",
        "não_utilizo_nenhuma_ferramenta_de_bi_no_trabalho",
        "mysql",
        "oracle",
        "sql_server",
        "amazon_aurora_ou_rds",
        "dynamodb",
        "coachdb",
        "cassandra",
        "mongodb",
        "mariadb",
        "datomic",
        "s3",
        "postgresql",
        "elasticsearch",
        "db2",
        "microsoft_access",
        "sqlite",
        "sybase",
        "firebase",
        "vertica",
        "redis",
        "neo4j",
        "google_bigquery",
        "google_firestore",
        "amazon_redshift",
        "amazon_athena",
        "snowflake",
        "hbase",
        "presto",
        "splunk",
        "sap_hana",
        "hive",
        "firebird",
        "cloud_(dia_a_dia)",
        "amazon_web_services_(aws)",
        "google_cloud_(gcp)",
        "azure_(microsoft)",
        "oracle_cloud",
        "ibm",
        "servidores_on_premise/não_utilizamos_cloud",
        "cloud_própria",
        "cloud_preferida",
        "sql",
        "r",
        "python",
        "c/c++/c#",
        ".net",
        "java",
        "julia",
        "sas/stata",
        "visual_basic/vba",
        "scala",
        "matlab",
        "rust",
        "php",
        "javascript",
        "não_utilizo_nenhuma_das_linguagens_listadas",
        "linguagem_mais_usada",
        "linguagem_preferida",
        "dados_relacionais_(estruturados_em_bancos_sql)",
        "dados_armazenados_em_bancos_nosql",
        "imagens",
        "textos/documentos",
        "vídeos",
        "áudios",
        "planilhas",
        "dados_georeferenciados",
        "3.g.1_falta_de_compreensão_dos_casos_de_uso",
        "3.g.2_falta_de_confiabilidade_das_saídas_(alucinação_dos_modelos)",
        "3.g.3_incerteza_em_relação_a_regulamentação",
        "3.g.4_preocupações_com_segurança_e_privacidade_de_dados",
        "3.g.5_retorno_sobre_investimento_(roi)_não_comprovado_de_ia_generativa",
        "3.g.6_dados_da_empresa_não_estão_prontos_para_uso_de_ia_generativa",
        "3.g.7_falta_de_expertise_ou_falta_de_recursos",
        "3.g.8_alta_direção_da_empresa_não_vê_valor_ou_não_vê_como_prioridade",
        "3.g.9_preocupações_com_propriedade_intelectual",
        "3.f.1_colaboradores_usando_ai_generativa_de_forma_independente_e_descentralizada",
        "3.f.2_direcionamento_centralizado_do_uso_de_ai_generativa",
        "3.f.3_desenvolvedores_utilizando_copilots",
        "3.f.4_ai_generativa_e_llms_para_melhorar_produtos_externos_para_os_clientes_finais",
        "3.f.5_ai_generativa_e_llms_para_melhorar_produtos_internos_para_os_colaboradores",
        "3.f.6_ia_generativa_e_llms_como_principal_frente_do_negócio",
        "3.f.7_ia_generativa_e_llms_não_é_prioridade",
        "3.f.8_não_sei_opinar_sobre_o_uso_de_ia_generativa_e_llms_na_empresa",
        "contratar_talentos",
        "reter_talentos",
        "convencer_a_empresa_a_aumentar_investimentos",
        "gestão_de_equipes_no_ambiente_remoto",
        "gestão_de_projetos_envolvendo_áreas_multidisciplinares",
        "organizar_as_informações_com_qualidade_e_confiabilidade",
        "processar_e_armazenar_um_alto_volume_de_dados",
        "gerar_valor_para_as_áreas_de_negócios",
        "desenvolver_e_manter_modelos_machine_learning_em_produção",
        "gerenciar_a_expectativa_das_áreas",
        "garantir_a_manutenção_dos_projetos_e_modelos_em_produção",
        "conseguir_levar_inovação_para_a_empresa",
        "garantir_(roi)_em_projetos_de_dados",
        "dividir_o_tempo_entre_entregas_técnicas_e_gestão",
        "pensar_na_visão_de_longo_prazo_de_dados",
        "organização_de_treinamentos_e_iniciativas",
        "atração,_seleção_e_contratação",
        "decisão_sobre_contratação_de_ferramentas",
        "gestor_da_equipe_de_engenharia_de_dados",
        "gestor_da_equipe_de_estudos,_relatórios",
        "gestor_da_equipe_de_inteligência_artificial_e_machine_learning",
        "apesar_de_ser_gestor_ainda_atuo_na_parte_técnica",
        "gestão_de_projetos_de_dados",
        "gestão_de_produtos_de_dados",
        "gestão_de_pessoas",
        "analytics_engineer",
        "engenharia_de_dados/data_engineer",
        "analista_de_dados/data_analyst",
        "cientista_de_dados/data_scientist",
        "database_administrator/dba",
        "analista_de_business_intelligence/bi",
        "arquiteto_de_dados/data_architect",
        "data_product_manager/dpm",
        "business_analyst",
        "ml_engineer/ai_engineer"
    ]

    # Removendo colunas
    df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

    return df

def salvar_dados(df, caminho):
    """Salva a versão limpa no staging"""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")
    print(f"\nArquivo limpo salvo em: {caminho}")

# ----------------------------
# 3. MAIN
# ----------------------------
def main():
    df = carregar_dados(RAW_PATH)
    print("Antes da limpeza:", df.shape)

    df_limpo = limpar_dados(df)
    print("Depois da limpeza:", df_limpo.shape)

    # Preview das 5 primeiras linhas
    print("\nPrévia dos dados limpos:")
    print(df_limpo.head().to_string(max_cols=10, max_rows=5))

    print("\nTipos de cada coluna:")
    for col, dtype in df_limpo.dtypes.items():
        print(f"{col}: {dtype}")

    colunas_para_amostra = [
        "satisfeito_atualmente",
        "remuneração/salário",
        "benefícios",
        "propósito_do_trabalho_e_da_empresa",
        "flexibilidade_de_trabalho_remoto",
        "ambiente_e_clima_de_trabalho",
        "oportunidade_de_aprendizado_e_trabalhar_com_referências",
        "oportunidades_de_crescimento",
        "maturidade_da_empresa_em_termos_de_tecnologia_e_dados",
        "relação_com_os_gestores_e_líderes",
        "reputação_que_a_empresa_tem_no_mercado",
        "gostaria_de_trabalhar_em_outra_área"

    ]

    print("\nAmostra aleatória de 10 linhas das colunas selecionadas:")
    print(df_limpo[colunas_para_amostra].sample(20, random_state=42).to_string(max_cols=15))

    salvar_dados(df_limpo, STAGING_PATH)

if __name__ == "__main__":
    main()
