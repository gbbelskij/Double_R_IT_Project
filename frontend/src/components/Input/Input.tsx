import { InputProps } from "./Input.props";
import classes from "./Input.module.css";
import { useState } from 'react';
import classNames from 'classnames';

import { FaCheck } from 'react-icons/fa6';
import { FaRegUserCircle } from 'react-icons/fa';
import { TbCalendarQuestion } from 'react-icons/tb';
import { AiOutlineMail } from 'react-icons/ai';
import { MdAccessTime } from 'react-icons/md';
import { LuKeyRound } from 'react-icons/lu';
import { LuEye } from 'react-icons/lu';
import { LuEyeOff } from 'react-icons/lu';

const defaultIcons = {
  text: <FaRegUserCircle className={classes.InputIcon}/>,
  date: <TbCalendarQuestion className={classes.InputIcon}/>,
  email: <AiOutlineMail className={classes.InputIcon}/>,
  number: <MdAccessTime className={classes.InputIcon}/>,
  password: <LuKeyRound className={classes.InputIcon}/>,
  checkbox: <FaCheck className={classes.InputIcon}/>,
};

import React from 'react';

const Input: React.FC<InputProps> = ({
  type,
  leftIcon,
  name,
  label,
  hideIcons = false,
  prefix,
  defaultValue,
  unit,
  ...props
}) => {
  const getDefaultIcon = () => {
    return defaultIcons[type] || null;
  };

  const [value, setValue] = useState(defaultValue || '');
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;

    if (type === 'number' && inputValue.length > 2) {
      return; 
    }

    setValue(inputValue);
  };

  const togglePasswordVisibility = () => {
    setIsPasswordVisible(!isPasswordVisible);
  };
  
  const typeMap: Record<string, string> = {
    date: 'date',
    email: 'email',
    number: 'number',
    text: 'text',
  };

  const renderPasswordInput = () => (
    <div className={classes.InputWrapper}>
      <input 
        className={classNames(classes.Input, classes.InputPassword)}
        id={name}
        name={name}
        type={isPasswordVisible ? 'text' : 'password'}
        value={value}
        onChange={handleChange}
        {...props}
      />
      <span className={classes.IconWrapper} onClick={togglePasswordVisibility}>
        {isPasswordVisible ? <LuEye className={classes.InputIcon}/> : <LuEyeOff className={classes.InputIcon}/>}
      </span>
    </div>
  );

  const renderCheckboxInput = () => (
    <div className={classNames(classes.Input, classes.InputCheckbox)}>
      <input 
        type="checkbox"
        id={name}
        name={name}
        checked={value === 'true'}
        onChange={(e) => setValue(e.target.checked ? 'true' : '')}
        {...props}
      />
      {value === 'true' && (
      <FaCheck className={classes.CheckIcon} />
    )}
      <label htmlFor={name}>{label}</label>
    </div>
  );

  return (
    <div className={classes.Inputs}>
      {type === 'checkbox' ? (
        renderCheckboxInput() 
      ) : (
        <>
          <label htmlFor={name} className={classes.InputsLabel}>{label}</label>
          <div className={classNames(classes.Input, classes[`InputItems_${type}`])}>
            {!hideIcons && (
                leftIcon ? leftIcon : getDefaultIcon()
            )}
            {type === 'number' && prefix && (
              <span className={classes.PrefixText}>{prefix}</span>
            )}  
            {type === 'password' ? (
              renderPasswordInput() 
            ) : (
              <div className={classes.InputsWrapper_default}>
                <input 
                  className={classNames(classes.Input, classes[`Input_${type}`])}
                  id={name}
                  name={name}
                  type={typeMap[type] || 'text'}
                  value={value}
                  onChange={handleChange}
                  placeholder={type === 'email' ? 'example@mail.com' : undefined}
                  min={0} 
                  max={99}
                  step={1}
                  {...props}
                />
                {type === 'number' && unit && value.trim() !== '' && (
                  <span className={classes.UnitText}>{unit}</span>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Input;