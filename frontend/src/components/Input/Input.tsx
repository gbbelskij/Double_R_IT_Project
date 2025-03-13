import { InputProps } from "./Input.props";
import classes from "./Input.module.css";

import { FaRegUserCircle, FaCheck } from 'react-icons/fa';
import { TbCalendarQuestion } from 'react-icons/tb';
import { AiOutlineMail } from 'react-icons/ai';
import { MdAccessTime } from 'react-icons/md';
import { LuKeyRound } from 'react-icons/lu';


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
  rightIcon,
  name,
  label,
  hideIcons = false,
  prefix,
  showEye = true,
  defaultValue,
  unit,
  ...props
}) => {
  const getDefaultIcon = () => {
    return defaultIcons[type] || null;
  };

  return (
    <div className={classes.Inputs}>
      <label htmlFor={name} className={classes.InputsLabel}>{label}</label>
      <div className={classes.InputItems}>
        {!hideIcons && (
            leftIcon ? leftIcon : getDefaultIcon()
        )}
        {type === 'number' && prefix && (
          <span className={classes.PrefixText}>{prefix}</span>
        )}
        <input className={classes.Input}
          id={name}
          name={name}
          type={type === 'password' && !showEye ? 'text' : type}
          defaultValue={type === 'email' ? defaultValue ?? 'example@mail.com' : undefined}
          {...props}
        />
        {!hideIcons && type === 'password' && (
            rightIcon ? rightIcon : getDefaultIcon()
        )}
      </div>
    </div>
  );
};

export default Input;