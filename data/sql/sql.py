def df2sql(df_name, df, conn):

    import sqlite3
    from sqlite3 import connect
    
    df.to_sql(df_name, conn, if_exists='append', index=False)


def sql2df(df_name, conn):

    import sqlite3
    from sqlite3 import connect

    dataframe = pd.read_sql("SELECT * FROM {0}".format(df_name), conn)

    return dataframe
