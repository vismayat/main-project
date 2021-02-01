import csv
import os

from flask import Flask,render_template,request,redirect
from flask.globals import session
from  DBConnection import Db

app = Flask(__name__)
app.secret_key="kk"




# ...............................admin...........................................................................................

@app.route('/u_panel')
def u_panel():
    if session['lin'] == "lin":
        return render_template("user/user_panel.html")
    else:
        return render_template("login.html")


@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login_pst',methods=['POST'])
def login_pst():
    usernam=request.form["email"]
    passwor=request.form["pass"]
    db=Db()
    qry=db.selectOne("SELECT * FROM login WHERE username='"+usernam+"' AND password='"+passwor+"' ")
    if qry is not None:
        if qry["type"]=="admin":
            session['lin']="lin"
            return ''' <script> alert('login success');window.location="/adm_home"; </script> '''

        elif(qry["type"]=="user"):
            q=db.selectOne("SELECT * FROM user WHERE user.login_id='"+str(qry["login_id"])+"'")
            if q is not None:
                session["userid"]=qry["login_id"]
                session['lin'] = "lin"
                return ''' <script>alert('login success');window.location="/u_panel"; </script> '''
            else:
                return ''' <script>alert('Invalid Username or Password');window.location="/"; </script> '''

    else:
        return ''' <script>alert('Invalid Username or Password');window.location="/"; </script> '''

@app.route('/adm_home')
def adm_home():
    if session['lin'] == "lin":
        return render_template("admin/AdminHome.html")
    else:
        return render_template("login.html")


@app.route('/adm_vcomt')
def adm_vcomt():
    if session['lin']=="lin":
        obj=Db()
        qry="select user.fname,user.email,complaint.* from user ,complaint where complaint.user_id=user.login_id"
        res=obj.select(qry)
        return render_template("admin/viewcomplaint.html",data=res)
    else:
        return render_template("login.html")






@app.route('/reply/<id>')
def reply(id):
    if session['lin'] == "lin":
        session['com']=id
        print(session)
        db = Db()
        qry="select * from complaint where c_id='"+id+"' "
        res=db.selectOne(qry)
        return render_template('admin/reply.html',data=res)
    else:
        return render_template("login.html")



@app.route('/adm_reply_post',methods=["post"])
def adm_reply_post():
    if session['lin'] == "lin":
        obj1=Db()
        reply=request.form['r']
        qry= "update complaint set reply='"+reply+"',reply_date=curdate(),status='replied' where c_id='"+str(session['com'])+"'"
        obj1.update(qry)
        return ''' <script> alert('replyed succesfully');window.location="/adm_vcomt"; </script> '''
    else:
        return render_template("login.html")






@app.route('/adm_vfbk')
def adm_vfbk():
    if session['lin'] == "lin":
        obj = Db()
        qry = "select user.fname,user.email,feedback.feedback,feedback.create_at from user,feedback where feedback.user_id=user.login_id"
        res = obj.select(qry)
        return render_template("admin/viewfeedbk.html",data=res)
    else:
        return render_template("login.html")





@app.route('/adm_vnoti')
def adm_vnoti():
   return render_template("admin/nofication.html")


@app.route('/adm_noty_post',methods=["post"])
def adm_noty_post():
    obj1=Db()
    Notification=request.form['fb']
    qry= "insert into notification values(null,'"+Notification+"',CURDATE()) "
    obj1.insert(qry)

    return ''' <script> alert('send succesfully');window.location="/adm_vnoti"; </script> '''





@app.route('/adm_vuser')
def adm_vuser():
    if session['lin'] == "lin":
        obj = Db()
        qry = "select * from user"
        res = obj.select(qry)
        return render_template("admin/viewuser.html", data=res)
    else:
        return render_template("login.html")

@app.route('/delt_user/<id>')
def delt_user(id):
    if session['lin'] == "lin":
        obj = Db()
        qry = "delete from user where user_id='" + id + "'"
        obj.delete(qry)
        return adm_vuser()
    else:
        return render_template("login.html")




@app.route('/adm_reply')
def adm_reply():
    if session['lin'] == "lin":
        return render_template("admin/reply.html")
    else:
        return render_template("login.html")


@app.route('/logout')
def logout():
    session['lin']="lout"
    return render_template("login.html")


# .................................user................................................................................................

@app.route('/userhome')
def userhome():
    return render_template("userhome.html")

#..........................................main......................................................................................


@app.route('/user_song')
def user_song():
    if session['lin'] == "lin":
        return render_template("user/checksong.html")
    else:
        return render_template("login.html")

@app.route('/song_post',methods=["post"])
def song_post():

    sng=request.files['song']
    sng.save("D:\\sound\\static\\a.wav")

    import librosa
    import numpy as np
    y, sr = librosa.load("D:\\sound\\static\\a.wav", mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    S, phase = librosa.magphase(librosa.stft(y))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    toappend=[]
    toappend.append(np.mean(chroma_stft))
    toappend.append(np.mean(spec_cent))
    toappend.append(np.mean(spec_bw))
    toappend.append(np.mean(rolloff))
    toappend.append(np.mean(zcr))




    for e in mfcc:
        toappend.append( np.mean(e))

    aatest= np.array([toappend])
    import pandas as pd

    a=pd.read_csv('D:\\sound\\data.csv')

    attributes= a.values[:,0:25]

    labels=a.values[:,25]


    print("aaa",attributes)

    print("bbb",labels)


    from sklearn.ensemble import RandomForestClassifier

    rnd=RandomForestClassifier()

    rnd.fit(attributes,labels)


    c=rnd.predict(np.array(aatest))

    print("predicted",c)


    db=Db()

    qry=" insert into song (song_id,user_id,song_file,status,created_at)values (null,'"+str( session["userid"])+"','"+sng.filename+"','"+str(c[0])+"',curdate())"
    res=db.insert(qry)
    print(res)







    return  render_template('/user/checksong.html',p=c,toappend=toappend)


    return "ok"

@app.route('/training')
def training():

    import librosa
    import pandas
    import numpy as np

    c=[]
    header = 'chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += ' mfcc'+str(i)
    header += ' label'
    header = header.split()

    file = open('D:\\sound\\data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock '.split()

    for g in genres:
        for filename in os.listdir("D:\\sound\\static\\data_set\\"+g):
            songname = "D:\\sound\\static\\data_set\\"+g+"\\"+filename

            aa=[]

            y, sr = librosa.load(songname, mono=True)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

            S, phase = librosa.magphase(librosa.stft(y))



            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = str(np.mean(chroma_stft)) +" "+str(np.mean(spec_cent)) +" "+ str(np.mean(spec_bw)) +" "+str(np.mean(rolloff)) +" "+str(np.mean(zcr))

            aa.append(np.mean(chroma_stft))
            aa.append(np.mean(spec_cent))
            aa.append(np.mean(spec_bw))
            aa.append(np.mean(rolloff))
            aa.append(np.mean(zcr))



            for e in mfcc:
                to_append += " "+str(np.mean(e))
                aa.append(np.mean(e))

            to_append +=  " "+g
            aa.append(g)

            file = open('D:\\sound\\data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


            c.append(aa)







    return  render_template('/admin/a.html',c=c)



#...........................................................................................

@app.route('/dataset_prfm')
def dataset_prfm():

    session["lin"]="lin"

    if session['lin'] == "lin":


        import pandas
        a=pandas.read_csv('D:\\sound\\data.csv')
        attributes= a.values[:,0:25]
        label=a.values[:,25]

        print(attributes)
        print(label)


        from sklearn.model_selection import  train_test_split

        X_train, X_test, y_train, y_test = train_test_split( attributes, label, test_size = 0.1, random_state = 42)

        from sklearn.ensemble import  RandomForestClassifier

        a=RandomForestClassifier()

        a.fit(X_train,y_train)


        predictedresult=a.predict(X_test)
        actualresult=y_test
        testdata=X_test

        l=len(testdata)

        from sklearn.metrics import accuracy_score

        sc=accuracy_score(actualresult,predictedresult)

        print(sc)

        return render_template("admin/matrix.html",p=predictedresult,a=actualresult,t=testdata,le=l,daccuracy=sc)

    else:
        return render_template("login.html")

@app.route('/dataset_prfm_dt')
def dataset_prfm_dt():

        session["lin"] = "lin"

        if session['lin'] == "lin":

            import pandas
            a = pandas.read_csv('D:\\sound\\data.csv')
            attributes = a.values[:, 0:25]
            label = a.values[:, 25]

            print(attributes)
            print(label)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.1, random_state=42)

            from sklearn.tree import DecisionTreeClassifier

            a = DecisionTreeClassifier()

            a.fit(X_train, y_train)

            predictedresult = a.predict(X_test)
            actualresult = y_test
            testdata = X_test

            l = len(testdata)

            from sklearn.metrics import accuracy_score

            sc = accuracy_score(actualresult, predictedresult)

            print(sc)

            return render_template("admin/matrix.html", p=predictedresult, a=actualresult, t=testdata, le=l,
                                   daccuracy=sc)

        else:
            return render_template("login.html")


#...............................................................................



@app.route('/dataset_prfm_lr')
def dataset_prfm_lr():

        session["lin"] = "lin"

        if session['lin'] == "lin":

            import pandas
            a = pandas.read_csv('D:\\sound\\data.csv')
            attributes = a.values[:, 0:25]
            label = a.values[:, 25]

            print(attributes)
            print(label)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.1, random_state=42)

            from sklearn.linear_model import LogisticRegression

            a = LogisticRegression()

            a.fit(X_train, y_train)

            predictedresult = a.predict(X_test)
            actualresult = y_test
            testdata = X_test

            l = len(testdata)

            from sklearn.metrics import accuracy_score

            sc = accuracy_score(actualresult, predictedresult)

            print(sc)

            return render_template("admin/matrix.html", p=predictedresult, a=actualresult, t=testdata, le=l,
                                   daccuracy=sc)

        else:
            return render_template("login.html")



@app.route('/dataset_prfm_nb')
def dataset_prfm_nb():

        session["lin"] = "lin"

        if session['lin'] == "lin":

            import pandas
            a = pandas.read_csv('D:\\sound\\data.csv')
            attributes = a.values[:, 0:25]
            label = a.values[:, 25]

            print(attributes)
            print(label)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.1, random_state=42)

            from sklearn.naive_bayes import GaussianNB

            a = GaussianNB()

            a.fit(X_train, y_train)

            predictedresult = a.predict(X_test)
            actualresult = y_test
            testdata = X_test

            l = len(testdata)

            from sklearn.metrics import accuracy_score

            sc = accuracy_score(actualresult, predictedresult)

            print(sc)

            return render_template("admin/matrix.html", p=predictedresult, a=actualresult, t=testdata, le=l,
                                   daccuracy=sc)

        else:
            return render_template("login.html")






@app.route('/user_fbk')
def user_fbk():
    if session['lin'] == "lin":
        return render_template("user/feedback.html")
    else:
        return render_template("login.html")


@app.route('/user_fdbk_post',methods=["post"])
def user_fdbk_post():
    if session['lin'] == "lin":
        obj1=Db()
        feedback=request.form['fb']
        qry= "insert into feedback (f_id,user_id,feedback,create_at)values (null,'"+str( session["userid"])+"','"+feedback+"',curdate())"
        obj1.insert(qry)

        return ''' <script> alert('send succesfully');window.location="/u_panel"; </script> '''
    else:
        return render_template("login.html")




@app.route('/user_vcomt')
def user_vcomt():
    if session['lin'] == "lin":
        obj = Db()
        x=session['userid']
        qry = "select subject,complaint,created_at,reply,reply_date from complaint where user_id='"+str(x)+"'"
        res = obj.select(qry)
        return render_template("user/viewcmplt.html",data=res)
    else:
        return render_template("login.html")

@app.route('/user_cmpt')
def user_cmpt():
    if session['lin'] == "lin":
        return render_template("user/complaints.html")
    else:
        return render_template("login.html")


@app.route('/user_cmplt_post',methods=["post"])
def user_cmplt_post():
    if session['lin'] == "lin":
        obj1=Db()
        Subject=request.form['subject']
        complaint=request.form['cmp']
        qry= "insert into complaint(user_id,subject,complaint,created_at,status) values('"+str( session["userid"])+"','"+Subject+"','"+complaint+"',curdate(),'pending')"
        obj1.insert(qry)

        return ''' <script> alert('send succesfully');window.location="/u_panel"; </script> '''
    else:
        return render_template("login.html")





@app.route('/user_noty')
def user_noty():
    obj = Db()
    qry = "select * from notification"
    res = obj.select(qry)
    return render_template("user/viewnotifi.html",data=res)


@app.route('/view_noti')
def view_noti():
    obj = Db()
    qry = "select * from notification"
    res = obj.select(qry)
    return render_template("admin/view_noti.html",data=res)


@app.route('/delt_noty/<id>')
def delt_noty(id):
    obj=Db()
    qry="delete from notification where n_id='"+id+"'"
    obj.delete(qry)
    return view_noti()



@app.route('/user_profile')
def user_profile():
    if session['lin'] == "lin":
        obj=Db()
        x=session['userid']
        qry="select fname,lname,gender,dob,email from user where login_id='"+str(x)+"' "
        res=obj.selectOne(qry)
        return render_template("user/profile.html",data=res)
    else:
        return render_template("login.html")




@app.route('/user_history')
def user_history():
    if session['lin'] == "lin":
        obj = Db()
        x = session['userid']
        qry = "select created_at,song_file,status,song_id from song where user_id='" + str(x) + "' "
        res = obj.select(qry)
        return render_template("user/history.html",data=res)
    else:
        return render_template("login.html")



@app.route('/d_history/<id>')
def d_history(id):
    if session['lin'] == "lin":
        obj1=Db()
        qry1="delete from song where song_id='"+id+"'"
        obj1.delete(qry1)
        return user_history()
    else:
        return render_template("login.html")


# .........................common............................................




@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/user_reg_post',methods=["post"])
def user_reg_post():
    print('KKKKKKKKKK')
    obj=Db()
    fname=request.form['firstname']
    lname=request.form['lastname']
    Gender=request.form['Gender']
    DOB=request.form['dob']
    Email=request.form['email']
    Password=request.form['paswd']
    confirmPassword=request.form['cpaswd']
    if confirmPassword==Password:
        qry1 = "insert into login(username,password,type)values('"+Email+"','"+confirmPassword+"','user')"
        res=obj.insert(qry1)
        qry2="insert into user(login_id,fname,lname,Gender,dob,email)values('"+str(res)+"','"+fname+"','"+lname+"','"+Gender+"','"+DOB+"','"+Email+"')"
        obj.insert(qry2)
        return ''' <script>alert('signup success');window.location="/"; </script> '''
    else:
        "error"
        return redirect('/')








@app.route('/cpswd')
def cpswd():
    if session['lin'] == "lin":
        return render_template("changepswd.html")
    else:
        return render_template("login.html")


@app.route('/pchange',methods=["post"])
def pchange():
    if session['lin'] == "lin":
        obj1=Db()
        crpswd=request.form['cpsw']
        newpswd=request.form['newpsw']
        cnfpswd = request.form['cpswd']

        if newpswd == cnfpswd:
            qry= "select password from login where login_id='"+str(session["userid"])+"'"
            res=obj1.selectOne(qry)
            if res is not None:
                if res['password']==crpswd:
                    q="update login set password='"+newpswd+"' where login_id='"+str( session["userid"])+"'"
                    obj1.update(q)
                    return userhome()
                else:
                    return '''<script>alert('incorrect current password');window.location='/cpswd';</script>'''
        else:
            return '''<script>alert(' passwords not matching ');window.location='/cpswd';</script>'''
    else:
        return render_template("login.html")







if __name__ == '__main__':
    app.run(debug=True)
