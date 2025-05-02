import React, { useEffect, useRef, useState } from "react";
import classNames from "classnames";

import { FaCaretDown, FaCaretUp } from "react-icons/fa";

import { useWindowSize } from "@hooks/useWindowSize";

import { SelectProps } from "./Select.props";
import { SelectOption } from "./Select.types";

import classes from "./Select.module.css";

const Select: React.FC<SelectProps> = ({
  options,
  name,
  label,
  icon,
  defaultValue,
  placeholder,
  register,
  error,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState<SelectOption | null>(
    defaultValue ? options.find((o) => o.value === defaultValue) || null : null
  );

  const { isSmallMobile } = useWindowSize();

  const iconSize = isSmallMobile ? 20 : 28;

  const selectRef = useRef<HTMLDivElement>(null);

  const { ref, onChange, ...rest } = register ? register(name) : {};

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        selectRef.current &&
        !selectRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (option: SelectOption) => {
    setSelected(option);
    setIsOpen(false);

    if (onChange) {
      onChange({ target: { name, value: option.value } });
    }
  };

  const Icon = icon;

  return (
    <div className={classes.Select} ref={selectRef}>
      <label
        className={classes.SelectField}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <div className={classes.SelectLabel}>{label}</div>

        <div
          className={classNames(classes.SelectWrapper, {
            [classes.SelectWrapperOpen]: isOpen,
            [classes.SelectWrapperWithError]: error,
          })}
        >
          {Icon && <Icon size={iconSize} />}

          <div
            className={classNames(classes.SelectValue, {
              [classes.SelectPlaceholder]: !selected,
            })}
          >
            <span className={classes.TruncatedText}>
              {selected ? selected.label : placeholder}
            </span>
          </div>

          {isOpen ? (
            <FaCaretUp size={iconSize} />
          ) : (
            <FaCaretDown size={iconSize} />
          )}
        </div>
      </label>

      {isOpen && (
        <ul className={classes.SelectOptions}>
          {options.map((option, index) => (
            <React.Fragment key={option.value}>
              <li
                className={classNames(classes.SelectOption, {
                  [classes.SelectOptionSelected]:
                    selected?.value === option.value,
                })}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelect(option);
                }}
                title={option.label}
              >
                <span
                  className={classNames(
                    classes.SelectOptionLabel,
                    classes.TruncatedText
                  )}
                >
                  {option.label}
                </span>
              </li>

              {index !== options.length - 1 && (
                <div className={classes.SelectDivider} />
              )}
            </React.Fragment>
          ))}
        </ul>
      )}

      <input
        type="hidden"
        name={name}
        value={selected?.value || ""}
        ref={ref}
        {...rest}
      />

      {error && !isOpen && (
        <p className={classes.SelectErrorText}>{error.message}</p>
      )}
    </div>
  );
};

export default Select;
