import React from "react";
import { FormProps } from "./Form.props";

import classes from "./Form.module.css";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import Button from "@components/Button/Button";
import { Link } from "react-router";
import classNames from "classnames";

const Form = ({
  id,
  children,
  title,
  additionalText,
  actionText = "Отправить",
  handleAction = () => {},
  isButtonDisabled = false,
  helperText,
  helperLinkText,
  helperLink,
  disableLogo,
  logoOffset,
  formClassName,
}: FormProps) => {
  const Wrapper = disableLogo ? React.Fragment : LogoContainer;

  return (
    <Wrapper logoOffset={logoOffset}>
      <div className={classes.Form}>
        <div className={classes.FormHeading}>
          {title && <h2 className={classes.FormHeadingText}>{title}</h2>}
          {additionalText && (
            <p className={classes.FormAdditionalText}>{additionalText}</p>
          )}
        </div>

        <form
          className={classNames(classes.FormForm, {
            [formClassName!]: formClassName,
          })}
          id={id}
          onSubmit={(e) => {
            e.preventDefault();
            handleAction();
          }}
        >
          {children}
        </form>

        <div className={classes.FormLower}>
          <Button
            color="inverse"
            size="large"
            type="submit"
            form={id}
            isFullWidth
            disabled={isButtonDisabled}
          >
            {actionText}
          </Button>

          <div className={classes.FormHelper}>
            {helperText && (
              <p className={classes.FormHelperText}>{helperText}</p>
            )}
            {helperLinkText && helperLink && (
              <Link to={helperLink} className={classes.FormHelperLink}>
                {helperLinkText}
              </Link>
            )}
          </div>
        </div>
      </div>
    </Wrapper>
  );
};

export default Form;
