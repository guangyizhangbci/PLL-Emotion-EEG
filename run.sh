#!/bin/bash


read -p "Enter the method name (string): " method_name
read -p "Enter if use label disambiguation (boolean): " confidence_boolean
read -p "Enter the partial label type (string): " partial_type_name

if [[ $method_name == "DNPL" ]]
then
  for i in 1 2 3 4 5
  do
    python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --run-idx $i
    wait
  done

elif [[ $method_name == "PRODEN" ]] || [[ $method_name == "CAVL" ]]
then
  for i in 1 2 3 4 5
  do
    if [[ $confidence_boolean == true ]]
    then
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --use-confidence --run-idx $i
      wait
    else
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --run-idx $i
      wait
    fi
  done

elif [[ $method_name == "LW" ]]
then
  read -p "Enter the loss name (string): " loss_name
  read -p "Enter the beta value (int): " beta_value

  for i in 1 2 3 4 5
  do
    if [[ $confidence_boolean == true ]]
    then
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --use-confidence --loss $loss_name --beta $beta_value --run-idx $i
      wait
    else
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --loss $loss_name --beta $beta_value --run-idx $i
      wait
    fi
  done

elif [[ $method_name == "PiCO" ]]
then
  read -p "Enter if use contrastive learning (boolean): " contrastive_boolean
  if [[ $contrastive_boolean == true ]]
  then
    gamme_value=0.5
  else
    gamme_value=0.0
  fi

  for i in 1 2 3 4 5
  do
    if [[ $confidence_boolean == true ]]
    then
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --use-confidence --gamma $gamme_value --run-idx $i
      wait
    else
      python3 /media/patrick/OS/SEED_V_code/PLL/main.py --optimizer 'sgd' --lr 0.01 --partial-type $partial_type_name --method $method_name --use-scheduler --gamma $gamme_value --run-idx $i
      wait
    fi
  done
    
else
  echo "Error!"
  exit
  
  
fi







  #

