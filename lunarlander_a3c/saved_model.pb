Х
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ющ

a2c_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*'
shared_namea2c_model/dense/kernel

*a2c_model/dense/kernel/Read/ReadVariableOpReadVariableOpa2c_model/dense/kernel*
_output_shapes
:	x*
dtype0

a2c_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namea2c_model/dense/bias
z
(a2c_model/dense/bias/Read/ReadVariableOpReadVariableOpa2c_model/dense/bias*
_output_shapes	
:*
dtype0

a2c_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*)
shared_namea2c_model/dense_1/kernel

,a2c_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpa2c_model/dense_1/kernel*
_output_shapes
:	x*
dtype0

a2c_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namea2c_model/dense_1/bias
~
*a2c_model/dense_1/bias/Read/ReadVariableOpReadVariableOpa2c_model/dense_1/bias*
_output_shapes	
:*
dtype0

a2c_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namea2c_model/dense_2/kernel

,a2c_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpa2c_model/dense_2/kernel* 
_output_shapes
:
*
dtype0

a2c_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namea2c_model/dense_2/bias
~
*a2c_model/dense_2/bias/Read/ReadVariableOpReadVariableOpa2c_model/dense_2/bias*
_output_shapes	
:*
dtype0

a2c_model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namea2c_model/dense_3/kernel

,a2c_model/dense_3/kernel/Read/ReadVariableOpReadVariableOpa2c_model/dense_3/kernel*
_output_shapes
:	*
dtype0

a2c_model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namea2c_model/dense_3/bias
}
*a2c_model/dense_3/bias/Read/ReadVariableOpReadVariableOpa2c_model/dense_3/bias*
_output_shapes
:*
dtype0

a2c_model/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namea2c_model/dense_4/kernel

,a2c_model/dense_4/kernel/Read/ReadVariableOpReadVariableOpa2c_model/dense_4/kernel*
_output_shapes
:	*
dtype0

a2c_model/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namea2c_model/dense_4/bias
}
*a2c_model/dense_4/bias/Read/ReadVariableOpReadVariableOpa2c_model/dense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

Adam/a2c_model/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*.
shared_nameAdam/a2c_model/dense/kernel/m

1Adam/a2c_model/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense/kernel/m*
_output_shapes
:	x*
dtype0

Adam/a2c_model/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/a2c_model/dense/bias/m

/Adam/a2c_model/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*0
shared_name!Adam/a2c_model/dense_1/kernel/m

3Adam/a2c_model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_1/kernel/m*
_output_shapes
:	x*
dtype0

Adam/a2c_model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_1/bias/m

1Adam/a2c_model/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/a2c_model/dense_2/kernel/m

3Adam/a2c_model/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/a2c_model/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_2/bias/m

1Adam/a2c_model/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_2/bias/m*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/a2c_model/dense_3/kernel/m

3Adam/a2c_model/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_3/kernel/m*
_output_shapes
:	*
dtype0

Adam/a2c_model/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_3/bias/m

1Adam/a2c_model/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/a2c_model/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/a2c_model/dense_4/kernel/m

3Adam/a2c_model/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_4/kernel/m*
_output_shapes
:	*
dtype0

Adam/a2c_model/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_4/bias/m

1Adam/a2c_model/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/a2c_model/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*.
shared_nameAdam/a2c_model/dense/kernel/v

1Adam/a2c_model/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense/kernel/v*
_output_shapes
:	x*
dtype0

Adam/a2c_model/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/a2c_model/dense/bias/v

/Adam/a2c_model/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*0
shared_name!Adam/a2c_model/dense_1/kernel/v

3Adam/a2c_model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_1/kernel/v*
_output_shapes
:	x*
dtype0

Adam/a2c_model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_1/bias/v

1Adam/a2c_model/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/a2c_model/dense_2/kernel/v

3Adam/a2c_model/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/a2c_model/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_2/bias/v

1Adam/a2c_model/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_2/bias/v*
_output_shapes	
:*
dtype0

Adam/a2c_model/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/a2c_model/dense_3/kernel/v

3Adam/a2c_model/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_3/kernel/v*
_output_shapes
:	*
dtype0

Adam/a2c_model/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_3/bias/v

1Adam/a2c_model/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/a2c_model/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/a2c_model/dense_4/kernel/v

3Adam/a2c_model/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_4/kernel/v*
_output_shapes
:	*
dtype0

Adam/a2c_model/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/a2c_model/dense_4/bias/v

1Adam/a2c_model/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/a2c_model/dense_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Т2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§1
valueѓ1B№1 Bщ1
Й
flatten

dense1

dense2

dense3

policy
	value
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
є
/iter

0beta_1

1beta_2
	2decay
3learning_ratemWmXmYmZm[m\#m]$m^)m_*m`vavbvcvdvevf#vg$vh)vi*vj
 
F
0
1
2
3
4
5
#6
$7
)8
*9
F
0
1
2
3
4
5
#6
$7
)8
*9
­
4non_trainable_variables
5layer_regularization_losses
regularization_losses
		variables
6layer_metrics

trainable_variables
7metrics

8layers
 
 
 
 
­
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
regularization_losses
	variables
trainable_variables
<metrics

=layers
TR
VARIABLE_VALUEa2c_model/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEa2c_model/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
regularization_losses
	variables
trainable_variables
Ametrics

Blayers
VT
VARIABLE_VALUEa2c_model/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEa2c_model/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics
regularization_losses
	variables
trainable_variables
Fmetrics

Glayers
VT
VARIABLE_VALUEa2c_model/dense_2/kernel(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEa2c_model/dense_2/bias&dense3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Hnon_trainable_variables
Ilayer_regularization_losses
Jlayer_metrics
regularization_losses
 	variables
!trainable_variables
Kmetrics

Llayers
VT
VARIABLE_VALUEa2c_model/dense_3/kernel(policy/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEa2c_model/dense_3/bias&policy/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
­
Mnon_trainable_variables
Nlayer_regularization_losses
Olayer_metrics
%regularization_losses
&	variables
'trainable_variables
Pmetrics

Qlayers
US
VARIABLE_VALUEa2c_model/dense_4/kernel'value/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEa2c_model/dense_4/bias%value/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
­
Rnon_trainable_variables
Slayer_regularization_losses
Tlayer_metrics
+regularization_losses
,	variables
-trainable_variables
Umetrics

Vlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEAdam/a2c_model/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/a2c_model/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_2/kernel/mDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_2/bias/mBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_3/kernel/mDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_3/bias/mBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/a2c_model/dense_4/kernel/mCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/a2c_model/dense_4/bias/mAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/a2c_model/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/a2c_model/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_2/kernel/vDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_2/bias/vBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/a2c_model/dense_3/kernel/vDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/a2c_model/dense_3/bias/vBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/a2c_model/dense_4/kernel/vCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/a2c_model/dense_4/bias/vAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1a2c_model/dense/kernela2c_model/dense/biasa2c_model/dense_3/kernela2c_model/dense_3/biasa2c_model/dense_1/kernela2c_model/dense_1/biasa2c_model/dense_2/kernela2c_model/dense_2/biasa2c_model/dense_4/kernela2c_model/dense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_40227913
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*a2c_model/dense/kernel/Read/ReadVariableOp(a2c_model/dense/bias/Read/ReadVariableOp,a2c_model/dense_1/kernel/Read/ReadVariableOp*a2c_model/dense_1/bias/Read/ReadVariableOp,a2c_model/dense_2/kernel/Read/ReadVariableOp*a2c_model/dense_2/bias/Read/ReadVariableOp,a2c_model/dense_3/kernel/Read/ReadVariableOp*a2c_model/dense_3/bias/Read/ReadVariableOp,a2c_model/dense_4/kernel/Read/ReadVariableOp*a2c_model/dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1Adam/a2c_model/dense/kernel/m/Read/ReadVariableOp/Adam/a2c_model/dense/bias/m/Read/ReadVariableOp3Adam/a2c_model/dense_1/kernel/m/Read/ReadVariableOp1Adam/a2c_model/dense_1/bias/m/Read/ReadVariableOp3Adam/a2c_model/dense_2/kernel/m/Read/ReadVariableOp1Adam/a2c_model/dense_2/bias/m/Read/ReadVariableOp3Adam/a2c_model/dense_3/kernel/m/Read/ReadVariableOp1Adam/a2c_model/dense_3/bias/m/Read/ReadVariableOp3Adam/a2c_model/dense_4/kernel/m/Read/ReadVariableOp1Adam/a2c_model/dense_4/bias/m/Read/ReadVariableOp1Adam/a2c_model/dense/kernel/v/Read/ReadVariableOp/Adam/a2c_model/dense/bias/v/Read/ReadVariableOp3Adam/a2c_model/dense_1/kernel/v/Read/ReadVariableOp1Adam/a2c_model/dense_1/bias/v/Read/ReadVariableOp3Adam/a2c_model/dense_2/kernel/v/Read/ReadVariableOp1Adam/a2c_model/dense_2/bias/v/Read/ReadVariableOp3Adam/a2c_model/dense_3/kernel/v/Read/ReadVariableOp1Adam/a2c_model/dense_3/bias/v/Read/ReadVariableOp3Adam/a2c_model/dense_4/kernel/v/Read/ReadVariableOp1Adam/a2c_model/dense_4/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_40228152
ю	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamea2c_model/dense/kernela2c_model/dense/biasa2c_model/dense_1/kernela2c_model/dense_1/biasa2c_model/dense_2/kernela2c_model/dense_2/biasa2c_model/dense_3/kernela2c_model/dense_3/biasa2c_model/dense_4/kernela2c_model/dense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/a2c_model/dense/kernel/mAdam/a2c_model/dense/bias/mAdam/a2c_model/dense_1/kernel/mAdam/a2c_model/dense_1/bias/mAdam/a2c_model/dense_2/kernel/mAdam/a2c_model/dense_2/bias/mAdam/a2c_model/dense_3/kernel/mAdam/a2c_model/dense_3/bias/mAdam/a2c_model/dense_4/kernel/mAdam/a2c_model/dense_4/bias/mAdam/a2c_model/dense/kernel/vAdam/a2c_model/dense/bias/vAdam/a2c_model/dense_1/kernel/vAdam/a2c_model/dense_1/bias/vAdam/a2c_model/dense_2/kernel/vAdam/a2c_model/dense_2/bias/vAdam/a2c_model/dense_3/kernel/vAdam/a2c_model/dense_3/bias/vAdam/a2c_model/dense_4/kernel/vAdam/a2c_model/dense_4/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_40228267ЊЫ
р

*__inference_dense_4_layer_call_fn_40228023

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_402278382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
­
E__inference_dense_2_layer_call_and_return_conditional_losses_40227975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
­
E__inference_dense_3_layer_call_and_return_conditional_losses_40227995

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
Ћ
C__inference_dense_layer_call_and_return_conditional_losses_40227935

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx:::O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
 


&__inference_signature_wrapper_40227913
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_402277022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
й.
№
#__inference__wrapped_model_40227702
input_12
.a2c_model_dense_matmul_readvariableop_resource3
/a2c_model_dense_biasadd_readvariableop_resource4
0a2c_model_dense_3_matmul_readvariableop_resource5
1a2c_model_dense_3_biasadd_readvariableop_resource4
0a2c_model_dense_1_matmul_readvariableop_resource5
1a2c_model_dense_1_biasadd_readvariableop_resource4
0a2c_model_dense_2_matmul_readvariableop_resource5
1a2c_model_dense_2_biasadd_readvariableop_resource4
0a2c_model_dense_4_matmul_readvariableop_resource5
1a2c_model_dense_4_biasadd_readvariableop_resource
identity

identity_1
a2c_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџx   2
a2c_model/flatten/Const
a2c_model/flatten/ReshapeReshapeinput_1 a2c_model/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2
a2c_model/flatten/ReshapeО
%a2c_model/dense/MatMul/ReadVariableOpReadVariableOp.a2c_model_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02'
%a2c_model/dense/MatMul/ReadVariableOpР
a2c_model/dense/MatMulMatMul"a2c_model/flatten/Reshape:output:0-a2c_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense/MatMulН
&a2c_model/dense/BiasAdd/ReadVariableOpReadVariableOp/a2c_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&a2c_model/dense/BiasAdd/ReadVariableOpТ
a2c_model/dense/BiasAddBiasAdd a2c_model/dense/MatMul:product:0.a2c_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense/BiasAdd
a2c_model/dense/ReluRelu a2c_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense/ReluФ
'a2c_model/dense_3/MatMul/ReadVariableOpReadVariableOp0a2c_model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'a2c_model/dense_3/MatMul/ReadVariableOpХ
a2c_model/dense_3/MatMulMatMul"a2c_model/dense/Relu:activations:0/a2c_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_3/MatMulТ
(a2c_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp1a2c_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(a2c_model/dense_3/BiasAdd/ReadVariableOpЩ
a2c_model/dense_3/BiasAddBiasAdd"a2c_model/dense_3/MatMul:product:00a2c_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_3/BiasAdd
a2c_model/dense_3/SoftmaxSoftmax"a2c_model/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_3/SoftmaxФ
'a2c_model/dense_1/MatMul/ReadVariableOpReadVariableOp0a2c_model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02)
'a2c_model/dense_1/MatMul/ReadVariableOpЦ
a2c_model/dense_1/MatMulMatMul"a2c_model/flatten/Reshape:output:0/a2c_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_1/MatMulУ
(a2c_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp1a2c_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(a2c_model/dense_1/BiasAdd/ReadVariableOpЪ
a2c_model/dense_1/BiasAddBiasAdd"a2c_model/dense_1/MatMul:product:00a2c_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_1/BiasAdd
a2c_model/dense_1/ReluRelu"a2c_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_1/ReluХ
'a2c_model/dense_2/MatMul/ReadVariableOpReadVariableOp0a2c_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'a2c_model/dense_2/MatMul/ReadVariableOpШ
a2c_model/dense_2/MatMulMatMul$a2c_model/dense_1/Relu:activations:0/a2c_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_2/MatMulУ
(a2c_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp1a2c_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(a2c_model/dense_2/BiasAdd/ReadVariableOpЪ
a2c_model/dense_2/BiasAddBiasAdd"a2c_model/dense_2/MatMul:product:00a2c_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_2/BiasAdd
a2c_model/dense_2/ReluRelu"a2c_model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_2/ReluФ
'a2c_model/dense_4/MatMul/ReadVariableOpReadVariableOp0a2c_model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'a2c_model/dense_4/MatMul/ReadVariableOpЧ
a2c_model/dense_4/MatMulMatMul$a2c_model/dense_2/Relu:activations:0/a2c_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_4/MatMulТ
(a2c_model/dense_4/BiasAdd/ReadVariableOpReadVariableOp1a2c_model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(a2c_model/dense_4/BiasAdd/ReadVariableOpЩ
a2c_model/dense_4/BiasAddBiasAdd"a2c_model/dense_4/MatMul:product:00a2c_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
a2c_model/dense_4/BiasAddw
IdentityIdentity#a2c_model/dense_3/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityz

Identity_1Identity"a2c_model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:џџџџџџџџџ:::::::::::T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Е
­
E__inference_dense_3_layer_call_and_return_conditional_losses_40227758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
Ћ
C__inference_dense_layer_call_and_return_conditional_losses_40227731

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx:::O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
Г
a
E__inference_flatten_layer_call_and_return_conditional_losses_40227712

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџx   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
­
E__inference_dense_1_layer_call_and_return_conditional_losses_40227955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx:::O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
б
­
E__inference_dense_4_layer_call_and_return_conditional_losses_40227838

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
­
E__inference_dense_1_layer_call_and_return_conditional_losses_40227785

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx:::O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
ш 
ї
G__inference_a2c_model_layer_call_and_return_conditional_losses_40227856
input_1
dense_40227742
dense_40227744
dense_3_40227769
dense_3_40227771
dense_1_40227796
dense_1_40227798
dense_2_40227823
dense_2_40227825
dense_4_40227849
dense_4_40227851
identity

identity_1Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallд
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_402277122
flatten/PartitionedCallІ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_40227742dense_40227744*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_402277312
dense/StatefulPartitionedCallЕ
dense_3/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_3_40227769dense_3_40227771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_402277582!
dense_3/StatefulPartitionedCallА
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_40227796dense_1_40227798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_402277852!
dense_1/StatefulPartitionedCallИ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_40227823dense_2_40227825*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_402278122!
dense_2/StatefulPartitionedCallЗ
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_4_40227849dense_4_40227851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_402278382!
dense_4/StatefulPartitionedCallЄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЈ

Identity_1Identity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:џџџџџџџџџ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

F
*__inference_flatten_layer_call_fn_40227924

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_402277122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
­
E__inference_dense_2_layer_call_and_return_conditional_losses_40227812

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ


,__inference_a2c_model_layer_call_fn_40227884
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_a2c_model_layer_call_and_return_conditional_losses_402278562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
N
о
!__inference__traced_save_40228152
file_prefix5
1savev2_a2c_model_dense_kernel_read_readvariableop3
/savev2_a2c_model_dense_bias_read_readvariableop7
3savev2_a2c_model_dense_1_kernel_read_readvariableop5
1savev2_a2c_model_dense_1_bias_read_readvariableop7
3savev2_a2c_model_dense_2_kernel_read_readvariableop5
1savev2_a2c_model_dense_2_bias_read_readvariableop7
3savev2_a2c_model_dense_3_kernel_read_readvariableop5
1savev2_a2c_model_dense_3_bias_read_readvariableop7
3savev2_a2c_model_dense_4_kernel_read_readvariableop5
1savev2_a2c_model_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_adam_a2c_model_dense_kernel_m_read_readvariableop:
6savev2_adam_a2c_model_dense_bias_m_read_readvariableop>
:savev2_adam_a2c_model_dense_1_kernel_m_read_readvariableop<
8savev2_adam_a2c_model_dense_1_bias_m_read_readvariableop>
:savev2_adam_a2c_model_dense_2_kernel_m_read_readvariableop<
8savev2_adam_a2c_model_dense_2_bias_m_read_readvariableop>
:savev2_adam_a2c_model_dense_3_kernel_m_read_readvariableop<
8savev2_adam_a2c_model_dense_3_bias_m_read_readvariableop>
:savev2_adam_a2c_model_dense_4_kernel_m_read_readvariableop<
8savev2_adam_a2c_model_dense_4_bias_m_read_readvariableop<
8savev2_adam_a2c_model_dense_kernel_v_read_readvariableop:
6savev2_adam_a2c_model_dense_bias_v_read_readvariableop>
:savev2_adam_a2c_model_dense_1_kernel_v_read_readvariableop<
8savev2_adam_a2c_model_dense_1_bias_v_read_readvariableop>
:savev2_adam_a2c_model_dense_2_kernel_v_read_readvariableop<
8savev2_adam_a2c_model_dense_2_bias_v_read_readvariableop>
:savev2_adam_a2c_model_dense_3_kernel_v_read_readvariableop<
8savev2_adam_a2c_model_dense_3_bias_v_read_readvariableop>
:savev2_adam_a2c_model_dense_4_kernel_v_read_readvariableop<
8savev2_adam_a2c_model_dense_4_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_83ebd3fda0f244289c7a470eda24edcb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameў
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*
valueB$B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB&policy/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesа
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesС
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_a2c_model_dense_kernel_read_readvariableop/savev2_a2c_model_dense_bias_read_readvariableop3savev2_a2c_model_dense_1_kernel_read_readvariableop1savev2_a2c_model_dense_1_bias_read_readvariableop3savev2_a2c_model_dense_2_kernel_read_readvariableop1savev2_a2c_model_dense_2_bias_read_readvariableop3savev2_a2c_model_dense_3_kernel_read_readvariableop1savev2_a2c_model_dense_3_bias_read_readvariableop3savev2_a2c_model_dense_4_kernel_read_readvariableop1savev2_a2c_model_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_adam_a2c_model_dense_kernel_m_read_readvariableop6savev2_adam_a2c_model_dense_bias_m_read_readvariableop:savev2_adam_a2c_model_dense_1_kernel_m_read_readvariableop8savev2_adam_a2c_model_dense_1_bias_m_read_readvariableop:savev2_adam_a2c_model_dense_2_kernel_m_read_readvariableop8savev2_adam_a2c_model_dense_2_bias_m_read_readvariableop:savev2_adam_a2c_model_dense_3_kernel_m_read_readvariableop8savev2_adam_a2c_model_dense_3_bias_m_read_readvariableop:savev2_adam_a2c_model_dense_4_kernel_m_read_readvariableop8savev2_adam_a2c_model_dense_4_bias_m_read_readvariableop8savev2_adam_a2c_model_dense_kernel_v_read_readvariableop6savev2_adam_a2c_model_dense_bias_v_read_readvariableop:savev2_adam_a2c_model_dense_1_kernel_v_read_readvariableop8savev2_adam_a2c_model_dense_1_bias_v_read_readvariableop:savev2_adam_a2c_model_dense_2_kernel_v_read_readvariableop8savev2_adam_a2c_model_dense_2_bias_v_read_readvariableop:savev2_adam_a2c_model_dense_3_kernel_v_read_readvariableop8savev2_adam_a2c_model_dense_3_bias_v_read_readvariableop:savev2_adam_a2c_model_dense_4_kernel_v_read_readvariableop8savev2_adam_a2c_model_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ў
_input_shapes
: :	x::	x::
::	::	:: : : : : :	x::	x::
::	::	::	x::	x::
::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	x:!

_output_shapes	
::%!

_output_shapes
:	x:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	x:!

_output_shapes	
::%!

_output_shapes
:	x:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	x:!

_output_shapes	
::%!

_output_shapes
:	x:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	: #

_output_shapes
::$

_output_shapes
: 
Г
a
E__inference_flatten_layer_call_and_return_conditional_losses_40227919

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџx   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџx2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
м
}
(__inference_dense_layer_call_fn_40227944

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_402277312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
р

*__inference_dense_3_layer_call_fn_40228004

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_402277582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

*__inference_dense_2_layer_call_fn_40227984

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_402278122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

А
$__inference__traced_restore_40228267
file_prefix+
'assignvariableop_a2c_model_dense_kernel+
'assignvariableop_1_a2c_model_dense_bias/
+assignvariableop_2_a2c_model_dense_1_kernel-
)assignvariableop_3_a2c_model_dense_1_bias/
+assignvariableop_4_a2c_model_dense_2_kernel-
)assignvariableop_5_a2c_model_dense_2_bias/
+assignvariableop_6_a2c_model_dense_3_kernel-
)assignvariableop_7_a2c_model_dense_3_bias/
+assignvariableop_8_a2c_model_dense_4_kernel-
)assignvariableop_9_a2c_model_dense_4_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate5
1assignvariableop_15_adam_a2c_model_dense_kernel_m3
/assignvariableop_16_adam_a2c_model_dense_bias_m7
3assignvariableop_17_adam_a2c_model_dense_1_kernel_m5
1assignvariableop_18_adam_a2c_model_dense_1_bias_m7
3assignvariableop_19_adam_a2c_model_dense_2_kernel_m5
1assignvariableop_20_adam_a2c_model_dense_2_bias_m7
3assignvariableop_21_adam_a2c_model_dense_3_kernel_m5
1assignvariableop_22_adam_a2c_model_dense_3_bias_m7
3assignvariableop_23_adam_a2c_model_dense_4_kernel_m5
1assignvariableop_24_adam_a2c_model_dense_4_bias_m5
1assignvariableop_25_adam_a2c_model_dense_kernel_v3
/assignvariableop_26_adam_a2c_model_dense_bias_v7
3assignvariableop_27_adam_a2c_model_dense_1_kernel_v5
1assignvariableop_28_adam_a2c_model_dense_1_bias_v7
3assignvariableop_29_adam_a2c_model_dense_2_kernel_v5
1assignvariableop_30_adam_a2c_model_dense_2_bias_v7
3assignvariableop_31_adam_a2c_model_dense_3_kernel_v5
1assignvariableop_32_adam_a2c_model_dense_3_bias_v7
3assignvariableop_33_adam_a2c_model_dense_4_kernel_v5
1assignvariableop_34_adam_a2c_model_dense_4_bias_v
identity_36ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*
valueB$B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB&policy/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDpolicy/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBpolicy/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvalue/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAvalue/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesж
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesт
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*І
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityІ
AssignVariableOpAssignVariableOp'assignvariableop_a2c_model_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ќ
AssignVariableOp_1AssignVariableOp'assignvariableop_1_a2c_model_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2А
AssignVariableOp_2AssignVariableOp+assignvariableop_2_a2c_model_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ў
AssignVariableOp_3AssignVariableOp)assignvariableop_3_a2c_model_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4А
AssignVariableOp_4AssignVariableOp+assignvariableop_4_a2c_model_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ў
AssignVariableOp_5AssignVariableOp)assignvariableop_5_a2c_model_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6А
AssignVariableOp_6AssignVariableOp+assignvariableop_6_a2c_model_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp)assignvariableop_7_a2c_model_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8А
AssignVariableOp_8AssignVariableOp+assignvariableop_8_a2c_model_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ў
AssignVariableOp_9AssignVariableOp)assignvariableop_9_a2c_model_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ї
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ў
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Й
AssignVariableOp_15AssignVariableOp1assignvariableop_15_adam_a2c_model_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16З
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_a2c_model_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Л
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_a2c_model_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Й
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_a2c_model_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Л
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_a2c_model_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Й
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_a2c_model_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Л
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_a2c_model_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Й
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_a2c_model_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Л
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_a2c_model_dense_4_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Й
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_a2c_model_dense_4_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Й
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_a2c_model_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26З
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_a2c_model_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Л
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_a2c_model_dense_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Й
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_a2c_model_dense_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Л
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_a2c_model_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Й
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_a2c_model_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Л
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_a2c_model_dense_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Й
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_a2c_model_dense_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Л
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_a2c_model_dense_4_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Й
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_a2c_model_dense_4_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpр
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35г
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*Ѓ
_input_shapes
: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
р

*__inference_dense_1_layer_call_fn_40227964

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_402277852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџx::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
б
­
E__inference_dense_4_layer_call_and_return_conditional_losses_40228014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*э
serving_defaultй
?
input_14
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџ<
output_20
StatefulPartitionedCall:1џџџџџџџџџtensorflow/serving/predict:м

flatten

dense1

dense2

dense3

policy
	value
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
*k&call_and_return_all_conditional_losses
l_default_save_signature
m__call__"ћ
_tf_keras_modelс{"class_name": "A2CModel", "name": "a2c_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "A2CModel"}}
т
regularization_losses
	variables
trainable_variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ь

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Ч
_tf_keras_layer­{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120]}}
№

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120]}}
№

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
*t&call_and_return_all_conditional_losses
u__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
ё

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
*v&call_and_return_all_conditional_losses
w__call__"Ь
_tf_keras_layerВ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
№

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}

/iter

0beta_1

1beta_2
	2decay
3learning_ratemWmXmYmZm[m\#m]$m^)m_*m`vavbvcvdvevf#vg$vh)vi*vj"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
Ъ
4non_trainable_variables
5layer_regularization_losses
regularization_losses
		variables
6layer_metrics

trainable_variables
7metrics

8layers
m__call__
l_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
regularization_losses
	variables
trainable_variables
<metrics

=layers
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
):'	x2a2c_model/dense/kernel
#:!2a2c_model/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
regularization_losses
	variables
trainable_variables
Ametrics

Blayers
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
+:)	x2a2c_model/dense_1/kernel
%:#2a2c_model/dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics
regularization_losses
	variables
trainable_variables
Fmetrics

Glayers
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,:*
2a2c_model/dense_2/kernel
%:#2a2c_model/dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Hnon_trainable_variables
Ilayer_regularization_losses
Jlayer_metrics
regularization_losses
 	variables
!trainable_variables
Kmetrics

Llayers
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
+:)	2a2c_model/dense_3/kernel
$:"2a2c_model/dense_3/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
Mnon_trainable_variables
Nlayer_regularization_losses
Olayer_metrics
%regularization_losses
&	variables
'trainable_variables
Pmetrics

Qlayers
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
+:)	2a2c_model/dense_4/kernel
$:"2a2c_model/dense_4/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
­
Rnon_trainable_variables
Slayer_regularization_losses
Tlayer_metrics
+regularization_losses
,	variables
-trainable_variables
Umetrics

Vlayers
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:,	x2Adam/a2c_model/dense/kernel/m
(:&2Adam/a2c_model/dense/bias/m
0:.	x2Adam/a2c_model/dense_1/kernel/m
*:(2Adam/a2c_model/dense_1/bias/m
1:/
2Adam/a2c_model/dense_2/kernel/m
*:(2Adam/a2c_model/dense_2/bias/m
0:.	2Adam/a2c_model/dense_3/kernel/m
):'2Adam/a2c_model/dense_3/bias/m
0:.	2Adam/a2c_model/dense_4/kernel/m
):'2Adam/a2c_model/dense_4/bias/m
.:,	x2Adam/a2c_model/dense/kernel/v
(:&2Adam/a2c_model/dense/bias/v
0:.	x2Adam/a2c_model/dense_1/kernel/v
*:(2Adam/a2c_model/dense_1/bias/v
1:/
2Adam/a2c_model/dense_2/kernel/v
*:(2Adam/a2c_model/dense_2/bias/v
0:.	2Adam/a2c_model/dense_3/kernel/v
):'2Adam/a2c_model/dense_3/bias/v
0:.	2Adam/a2c_model/dense_4/kernel/v
):'2Adam/a2c_model/dense_4/bias/v
2
G__inference_a2c_model_layer_call_and_return_conditional_losses_40227856Х
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
х2т
#__inference__wrapped_model_40227702К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
љ2і
,__inference_a2c_model_layer_call_fn_40227884Х
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_40227919Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_layer_call_fn_40227924Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_40227935Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_layer_call_fn_40227944Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_40227955Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_1_layer_call_fn_40227964Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_40227975Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_2_layer_call_fn_40227984Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_3_layer_call_and_return_conditional_losses_40227995Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_3_layer_call_fn_40228004Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_4_layer_call_and_return_conditional_losses_40228014Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_4_layer_call_fn_40228023Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5B3
&__inference_signature_wrapper_40227913input_1Я
#__inference__wrapped_model_40227702Ї
#$)*4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
Њ "cЊ`
.
output_1"
output_1џџџџџџџџџ
.
output_2"
output_2џџџџџџџџџл
G__inference_a2c_model_layer_call_and_return_conditional_losses_40227856
#$)*4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
Њ "KЂH
AЂ>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 В
,__inference_a2c_model_layer_call_fn_40227884
#$)*4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
Њ "=Ђ:

0џџџџџџџџџ

1џџџџџџџџџІ
E__inference_dense_1_layer_call_and_return_conditional_losses_40227955]/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_dense_1_layer_call_fn_40227964P/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџЇ
E__inference_dense_2_layer_call_and_return_conditional_losses_40227975^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dense_2_layer_call_fn_40227984Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_dense_3_layer_call_and_return_conditional_losses_40227995]#$0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
*__inference_dense_3_layer_call_fn_40228004P#$0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_dense_4_layer_call_and_return_conditional_losses_40228014])*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
*__inference_dense_4_layer_call_fn_40228023P)*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЄ
C__inference_dense_layer_call_and_return_conditional_losses_40227935]/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "&Ђ#

0џџџџџџџџџ
 |
(__inference_dense_layer_call_fn_40227944P/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџЅ
E__inference_flatten_layer_call_and_return_conditional_losses_40227919\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџx
 }
*__inference_flatten_layer_call_fn_40227924O3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџxн
&__inference_signature_wrapper_40227913В
#$)*?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџ"cЊ`
.
output_1"
output_1џџџџџџџџџ
.
output_2"
output_2џџџџџџџџџ