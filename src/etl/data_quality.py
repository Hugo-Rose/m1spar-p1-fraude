"""
Contrôle qualité Great Expectations sur le Silver P1 Fraude.
Lève ValueError si le taux de succès < 90%.
"""
from pyspark.sql import DataFrame


def run_expectations_p1(silver_df: DataFrame,
                        sample_size: int = 100_000) -> dict:
    """
    Valide la qualité du Silver P1 Fraude via Great Expectations.
    Lève une ValueError si le taux de succès < 90%.

    Expectations :
      - transaction_id  : non null, unique
      - amount          : entre 0.01 et 49 999
      - is_fraud        : moyenne entre 12% et 18%
      - velocity_1h     : entre 0 et 50
    """
    try:
        import great_expectations as ge
    except ImportError:
        raise ImportError(
            "great-expectations non installé. "
            "Lancez : pip install great-expectations==0.17.23"
        )

    df_ge = ge.from_pandas(
        silver_df.limit(sample_size).toPandas()
    )

    # Intégrité
    df_ge.expect_column_values_to_not_be_null("transaction_id")
    df_ge.expect_column_values_to_be_unique("transaction_id")

    # Montants (min=0.01, max=49 999)
    df_ge.expect_column_values_to_be_between(
        "amount", min_value=0.01, max_value=49_999)

    # Taux de fraude stable (12–18 %)
    df_ge.expect_column_mean_to_be_between(
        "is_fraud", min_value=0.12, max_value=0.18)

    # Vélocité plausible
    if "velocity_1h" in silver_df.columns:
        df_ge.expect_column_values_to_be_between(
            "velocity_1h", min_value=0, max_value=50)

    results = df_ge.validate()
    success  = results["statistics"]["success_percent"]
    print(f"Great Expectations : {success:.1f}% de succès")

    if success < 90:
        raise ValueError(f"Qualité insuffisante : {success:.1f}% < 90%")

    return results.to_json_dict()
