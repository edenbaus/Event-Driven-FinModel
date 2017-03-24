import MySQLdb as mdb
import pandas as pd
import os
import datetime as dt


def db_send(df):
    db_host = 'localhost'
    db_user = 'scottdb'
    db_pass = 'nycdsa'
    db_name = 'securities'
    con = mdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_name)

    df_ = df

    with con:
        cur = con.cursor()
        #cur.execute("SELECT * FROM stock_data;")
        for i in range(2286):
            tick_r, clos_r, dat_r = df_.ix[i]
            insert_str = "(" + "'" +tick_r+ "'"+ ", " + str(clos_r) + ", STR_TO_DATE(" + "'" + str(dat_r) + "'" + ",'%c/%e/%y'));"
            cur.execute( "INSERT INTO stock_data VALUES " + insert_str )
            #cur.execute(sql-insert)
        #d = cur.fetchall()
        print 'INSERTING: ' + str(tick_r)

            
pricedir = '../data/Price/' #directory
for file_ in set(os.listdir(pricedir)):
    try:
        g = len(file_)
        ticker_ = str(file_).split(".")[0]
        if g < 10:
            data_ = pd.read_csv(pricedir + file_)
            data_.columns = ['date', 'close']
            data_['ticker'] = ticker_
            cols = ['ticker', 'close', 'date']
            data_ = data_[cols]
            db_send(data_)
            #print data_
    except Exception, e:
        print 'error for %s' % (file_)
        print e
        continue
