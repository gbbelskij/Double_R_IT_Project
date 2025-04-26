import { ProfileFormProps } from "./ProfileForm.props";
import classNames from "classnames";

import Form from "@components/Form/Form";
import Button from "@components/Button/Button";

import classes from "./ProfileForm.module.css";
import { useWindowSize } from "@hooks/useWindowSize";

const ProfileForm: React.FC<ProfileFormProps> = ({
  id = "profile-form",
  children,
  handleAction = () => {},
  handleSurvey = () => {},
  handleExit = () => {},
  handleDelete = () => {},
  isButtonDisabled = false,
}) => {
  const { isMobile } = useWindowSize();

  const buttonSize = isMobile ? "medium" : "large";

  return (
    <div className={classes.ProfileForm}>
      <div className={classes.ProfileFormHeading}>
        <h2 className={classes.ProfileFormHeadingText}>
          Редактирование профиля
        </h2>

        <p className={classes.ProfileFormAdditionalText}>
          Выберите данные для изменения
        </p>
      </div>

      <Form
        plain
        id={id}
        formClassName={classNames(classes.ProfileFormForm)}
        handleAction={handleAction}
      >
        {children}
      </Form>

      <div className={classes.ProfileFormButtons}>
        <Button
          color="green"
          size={buttonSize}
          type="submit"
          form={id}
          isFullWidth
          disabled={isButtonDisabled}
        >
          Применить изменения
        </Button>

        <Button
          color="default"
          size={buttonSize}
          isFullWidth
          onClick={handleSurvey}
        >
          Пройти опрос повторно
        </Button>

        <div className={classes.ProfileFormSecondaryButtons}>
          <Button
            color="yellow"
            size={buttonSize}
            type="submit"
            isFullWidth
            onClick={handleExit}
          >
            Выйти из аккаунта
          </Button>

          <Button
            color="red"
            size={buttonSize}
            type="submit"
            isFullWidth
            onClick={handleDelete}
          >
            Удалить аккаунт
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ProfileForm;
