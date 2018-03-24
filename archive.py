'''
   x=np.copy(xnu)
   x[0,4]=abs(x[0,4]-ball[0])
   x[0,5] = abs(x[0,5] - ball[1])
   A, B = CalculateAB(x, model)  # we get A & B by deriving the Emulator
   K = LqrFhD2(A, B, Q, R,30)
   x = np.copy(np.transpose(x[0, :6]))
   uk = np.copy(-np.matmul(K,np.transpose(xnu[0,:6])))   # to reduce volume of action because of linearization
   uk = np.transpose(uk)*0.01
   #Mse1 = np.matmul((np.matmul(np.transpose(x), Q)), x)
   #Mse2 = np.matmul((np.matmul(uk, R)), np.transpose(uk))
   #print "MSE: " +str(Mse1+Mse2)
   '''

# model.fit(input, target, batch_size=10, epochs=100, verbose=2)
