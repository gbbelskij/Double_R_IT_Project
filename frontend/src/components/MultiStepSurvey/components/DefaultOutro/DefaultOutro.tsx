import { forwardRef } from "react";

import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";

import { Outro } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./DefaultOutro.module.css";

const DefaultOutro = forwardRef(({ onExit }, ref) => {
  const { isMobile } = useWindowSize();

  return (
    <LogoContainer>
      <div className={classes.DefaultOutro} ref={ref}>
        <div className={classes.DefaultOutroSection}>
          <h2 className={classes.DefaultOutroHeading}>
            Спасибо за ваши ответы!{" "}
          </h2>

          <p className={classes.DefaultOutroText}>
            Мы обработаем результаты и предложим вам лучшие курсы для успешного
            старта!
          </p>
        </div>

        <Button
          size={isMobile ? "medium" : "large"}
          color="inverse"
          isFullWidth
          onClick={() => onExit()}
        >
          Войти в аккаунт
        </Button>
      </div>
    </LogoContainer>
  );
}) as Outro;

export default DefaultOutro;
