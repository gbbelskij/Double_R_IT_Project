import { useNavigate } from "react-router";
import useWindowWidth from "@hooks/useWindowWidth";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";

import classes from "./DefaultOutro.module.css";

const DefaultOutro: React.FC = () => {
  const navigate = useNavigate();
  const windowWidth = useWindowWidth();

  return (
    <LogoContainer>
      <div className={classes.DefaultOutro}>
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
          size={windowWidth >= 768 ? "large" : "medium"}
          color="inverse"
          isFullWidth
          onClick={() => navigate("/login")}
        >
          Войти в аккаунт
        </Button>
      </div>
    </LogoContainer>
  );
};

export default DefaultOutro;
