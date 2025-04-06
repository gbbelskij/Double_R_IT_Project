import { useNavigate } from "react-router";

import Button from "@components/Button/Button";

import classes from "./DefaultOutro.module.css";

const DefaultOutro: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className={classes.DefaultOutro}>
      <div className={classes.DefaultOutroSection}>
        <h2 className={classes.DefaultOutroHeading}>
          Спасибо за ваши ответы!{" "}
        </h2>

        <p>
          Мы обработаем результаты и предложим вам лучшие курсы для успешного
          старта!
        </p>
      </div>

      <Button
        size="large"
        color="inverse"
        isFullWidth
        onClick={() => navigate("/login")}
      >
        Войти в аккаунт
      </Button>
    </div>
  );
};

export default DefaultOutro;
