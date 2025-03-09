import { InputProps } from "./Input.props";

import { FaRegUserCircle, FaCheck } from 'react-icons/fa';
import { TbCalendarQuestion } from 'react-icons/tb';
import { AiOutlineMail } from 'react-icons/ai';
import { MdAccessTime } from 'react-icons/md';
import { LuKeyRound } from 'react-icons/lu';


const defaultIcons = {
  text: <FaRegUserCircle />,
  date: <TbCalendarQuestion />,
  email: <AiOutlineMail />,
  number: <MdAccessTime />,
  password: <LuKeyRound />,
  checkbox: <FaCheck />,
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
  ...props
}) => {
  const getDefaultIcon = () => {
    return defaultIcons[type] || null;
  };

  return (
    <div>
      <label htmlFor={name}>{label}</label>
      <div>
        {!hideIcons && (
            leftIcon ? leftIcon : getDefaultIcon()
        )}
        {type === 'number' && prefix && (
          <span>{prefix}</span>
        )}
        <input
          id={name}
          name={name}
          type={type === 'password' && !showEye ? 'text' : type}
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