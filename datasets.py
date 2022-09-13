def normalized(x,y):
    xtrain, x_test, ytrain, y_test = None,None,None,None
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y, shuffle=True)
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    return x.numpy(), x_test.numpy(), y.numpy(), y_test.numpy(), scaler
def getuniquevalues(columnname):
    values={}
    i=0
    for entry in df[columnname]:
        if entry not in values:
            values[entry]=i
            i+=1
    return values
def get_dataset(name):
    datasets=[
        "KDD Cup 1999",                            #0
        "Microsoft Challenge BIG 2015"             #1
    ]
    #0
    if name==datasets[0]:
        path="../../Data/"
        attacks_types = {
            'normal': 'normal','back': 'dos','buffer_overflow': 'u2r','ftp_write': 'r2l','guess_passwd': 'r2l',
        'imap': 'r2l','ipsweep': 'probe','land': 'dos','loadmodule': 'u2r','multihop': 'r2l','neptune': 'dos',
        'nmap': 'probe','perl': 'u2r','phf': 'r2l','pod': 'dos','portsweep': 'probe','rootkit': 'u2r','satan': 'probe',
        'smurf': 'dos','spy': 'r2l','teardrop': 'dos','warezclient': 'r2l','warezmaster': 'r2l',
        }
        cols ="""duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,
        urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,
        num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,
        count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,
        diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,
        dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,
        dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"""
        
        columns =[]
        for c in cols.split(','):
            if(c.strip()):
                columns.append(c.strip())
        print(len(columns))
        columns.append('target')
        print(len(columns))

        attack_categories=["dos","u2r","r2l",'probe','normal']
        df = pd.read_csv(path+"kddcup.data_10_percent.gz", names = columns)
        df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
        del df['target']
        df.head()
        num_cols = df._get_numeric_data().columns
        
        cate_cols = list(set(df.columns)-set(num_cols))
        cate_cols.remove('Attack Type')
        for col in cate_cols:
            df[col]=df[col].map(getuniquevalues(col))
        data=df.to_numpy()
        Y=df['Attack Type'].map(getuniquevalues('Attack Type'))
        Y=Y.to_numpy()
        X=data[:,:-1]
        print(Y.shape)
        print(X.shape)
        print(getuniquevalues('Attack Type'))
        xtrain, x_test, ytrain, y_test,scaler= normalized(X,Y)
    #1
    if name==datasets[1]:
        path="../../Data/malware-classification/"
        map={}
        mapping=pd.read_csv(path + "trainLabels.csv")
        Y=mapping["Class"].to_numpy()
        for i in range(0,len(Y)):
            map[mapping["Id"][i]]=mapping["Class"][i]-1
        byte_features=pd.read_csv(path+"result.csv")
        byte_features['ID']  = byte_features['ID'].str.split('.').str[0]
        byte_features.head(3)
        byte_features['ID']=byte_features['ID'].map(map)
        data=byte_features.to_numpy()
        X=data[:,1:]
        Y=data[:,0]
        xtrain, x_test, ytrain, y_test,scaler= normalized(X,Y)
    return xtrain,x_test,ytrain,y_test