import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle



#Clean train dataset
def full_clean(df):
    #Create features
    df = create_fraud(df)
    df = create_usornot(df)
    df = create_time_diff(df)
    df = create_ticket_types(df)
    df = create_fraud_zone(df)
    
    #Drop fields not being used and fill empty records
    df = column_drop(df)
    df = fill_na(df,0)
    return df


#Clean incoming api data
def full_clean_incoming_data(df):
    #Create features
    df = create_usornot(df)
    df = create_time_diff(df)
    df = create_ticket_types(df)
    df = create_fraud_zone(df)
    
    #Drop fields not being used and fill empty records
    df = column_drop_incoming(df)
    df = fill_na(df,0)
    return df



#Create column indicating fraud or not
def create_fraud(df):
    df ['fraud'] = df['acct_type'].str.contains('fraud')
    return df


#Create indicating US or not US
def create_usornot(df):
    df['usornot']=(df.country=='US')
    return df

#Create timedelta fields for model
def create_time_diff(df):
    df['event_created']=pd.to_datetime(df['event_created'],unit='s')
    df['event_end']=pd.to_datetime(df['event_end'],unit='s')
    df['event_start']=pd.to_datetime(df['event_start'],unit='s')
    df['event_published']=pd.to_datetime(df['event_published'],unit='s')
    
    df['pub_start_diff']=(df.event_start - df.event_published).dt.days
    df['start_end_diff']=(df.event_end - df.event_start).dt.days
    df['created_pub_diff']=(df.event_published - df.event_created).dt.days
    return df


#Create ticket type features for model
def create_ticket_types(df):
    numevents=[]
    maxcostevent=[]
    avgcostevent=[]
    total=[]
    sold=[]
    diff=[]
    cost=[]
    
    for all_tickets in df['ticket_types']:
        
        if (len(all_tickets) == 0): #Check if ticket_types is empty
            maxcostevent.append(0)
            numevents.append(0)
            avgcostevent.append(0)
            total.append(0)
            sold.append(0)
            diff.append(0)
            cost.append(0)
        else:
            if 'cost' in all_tickets[0].keys(): #Check if ticket_types contains cost information
                maxcostevent.append(max([ticket['cost'] for ticket in all_tickets]))
                numevents.append(len([ticket['cost'] for ticket in all_tickets]))
                avgcostevent.append(sum([ticket['cost'] for ticket in all_tickets])/len([ticket['cost'] for ticket in all_tickets]))
                cost.append(sum([ticket['cost'] for ticket in all_tickets]))
            else:
                maxcostevent.append(0)
                numevents.append(0)
                avgcostevent.append(0)
                cost.append(0)
            if 'quantity_total' in all_tickets[0].keys(): #Check if ticket_types contains quantity total information
                total.append(sum([ticket['quantity_total'] for ticket in all_tickets]))
            else:
                total.append(0)
            if 'quantity_sold' in all_tickets[0].keys(): #Check if ticket_types contains quantity sold information
                sold.append(sum([ticket['quantity_sold'] for ticket in all_tickets]))
            else:
                sold.append(0)
            if ('quantity_total' in all_tickets[0].keys()) & ('quantity_sold' in all_tickets[0].keys()): #Check if ticket_types contains quantity sold and quantity total information
                diff.append(sum([ticket['quantity_total']-ticket['quantity_sold'] for ticket in all_tickets]))
            else:
                diff.append(0)
            
    
    
    
    df['ticket_numevents']=numevents
    df['ticket_maxcostevent']=maxcostevent
    df['ticket_avgcostevent']=avgcostevent
    df['ticket_total']=total
    df['ticket_sold']=sold
    df['ticket_diff']=diff
    df['ticket_cost']=cost

    return df

# Creating a Danger Zone
def create_fraud_zone(df):
   df['fraud_zone']= (df['venue_longitude'] > -50)& (df['venue_longitude'] <0) & (df['venue_latitude'] > 20) & (df['venue_latitude'] <46) \
   | (df['venue_longitude'] > 5)& (df['venue_longitude'] <50) & (df['venue_latitude'] > -40) & (df['venue_latitude'] <20) | \
   (df['venue_longitude'] > 50)& (df['venue_longitude'] <135) & (df['venue_latitude'] > -10)& (df['venue_latitude'] <35)
   return df

#Fill blank fields in model
def fill_na(df,value):
    df = df.fillna(value)
    return df



#Drop columns not used in model from api data
def column_drop_incoming(df): #third line where we have gts, are not in incoming data from API
    df = df.drop(['currency','description','email_domain','listed','ticket_types'\
              ,'name','org_desc','org_name','payee_name','payout_type','previous_payouts'\
              ,'venue_address','venue_country','venue_name','venue_state','has_header'\
              ,'event_created','event_end','event_published','event_start','user_created','country'], axis =1)
    return df


#Drop columns not used in model from training data
def column_drop(df): 
    df = df.drop(['acct_type','currency','description','email_domain','listed','ticket_types'\
              ,'name','org_desc','org_name','payee_name','payout_type','previous_payouts'\
              ,'gts','num_order','num_payouts','sale_duration2','approx_payout_date'\
              ,'venue_address','venue_country','venue_name','venue_state','has_header'\
              ,'event_created','event_end','event_published','event_start','user_created','country'], axis =1)

    return df

# Create pickled model from training data
# Need to run only once when you build model or make changes to it
def create_pickled_model(df):
    df=full_clean(df)
    X,y=X_and_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    rfc=RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rfc.fit(X_train,y_train)
    filename = 'finalized_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))
    return None



#Create confusion matrix
def confusion_mat(y_true,y_pred):
    tn, fp, fn, tp=confusion_matrix(y_true, y_pred).ravel()
    return np.array(([tp,fp],[fn,tn]))

#Get X and ys from dataframe
def X_and_y(df):
    y = df.pop('fraud').values
    X = df.values
    return X , y
